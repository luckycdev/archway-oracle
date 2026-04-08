import logging
import math
import os
import re
import time
from threading import Lock, Thread

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from camera_get_cams import fetch_cameras


load_dotenv()

DEFAULT_STREAM_SOURCE = os.getenv("DEFAULT_STREAM_SOURCE", "0")
DEFAULT_CAMERA_NAME = os.getenv("DEFAULT_CAMERA_NAME", "Default Camera")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo_stuff/yolo26n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.15"))
YOLO_CLASS_IDS = [int(item.strip()) for item in os.getenv("YOLO_CLASS_IDS", "2,3,5,7").split(",") if item.strip()]
MIN_BOX_BOTTOM_Y_RATIO = float(os.getenv("MIN_BOX_BOTTOM_Y_RATIO", "0.3"))
WORKER_IDLE_TIMEOUT_SECONDS = int(os.getenv("WORKER_IDLE_TIMEOUT_SECONDS", "10"))
MOVEMENT_STOPPED_THRESHOLD_PIXELS = float(os.getenv("MOVEMENT_STOPPED_THRESHOLD_PIXELS", "0.5"))
MOVEMENT_SLOW_THRESHOLD_PIXELS = float(os.getenv("MOVEMENT_SLOW_THRESHOLD_PIXELS", "4.0"))
MOVEMENT_MAX_MATCH_DISTANCE_PIXELS = float(os.getenv("MOVEMENT_MAX_MATCH_DISTANCE_PIXELS", "100.0"))
ROAD_MASK_STALE_SECONDS = float(os.getenv("ROAD_MASK_STALE_SECONDS", "5.0"))
ROAD_MASK_WARMUP_SECONDS = float(os.getenv("ROAD_MASK_WARMUP_SECONDS", "0.5"))
ROAD_MASK_FADE_MULTIPLIER = float(os.getenv("ROAD_MASK_FADE_MULTIPLIER", "0.95"))
ROAD_MASK_MIN_VALUE = int(os.getenv("ROAD_MASK_MIN_VALUE", "5"))
ROAD_LEARNING_MIN_FRAME_RATIO = float(os.getenv("ROAD_LEARNING_MIN_FRAME_RATIO", "0.05"))
FPS_SMOOTHING_ALPHA = float(os.getenv("FPS_SMOOTHING_ALPHA", "0.2"))
COVERAGE_SMOOTHING_ALPHA = float(os.getenv("COVERAGE_SMOOTHING_ALPHA", "0.2"))
TRAFFIC_BASELINE_COVERAGE = float(os.getenv("TRAFFIC_BASELINE_COVERAGE", "6.5"))
TRAFFIC_LIGHT_MAX = float(os.getenv("TRAFFIC_LIGHT_MAX", "2.5"))
TRAFFIC_MODERATE_MAX = float(os.getenv("TRAFFIC_MODERATE_MAX", "5.0"))
TRAFFIC_HEAVY_MAX = float(os.getenv("TRAFFIC_HEAVY_MAX", "7.5"))
PREDICTION_MAX_VEHICLES = float(os.getenv("PREDICTION_MAX_VEHICLES", "2500"))
WEATHER_PENALTY_SNOW = float(os.getenv("WEATHER_PENALTY_SNOW", "1.5"))
WEATHER_PENALTY_RAIN = float(os.getenv("WEATHER_PENALTY_RAIN", "0.8"))
WEATHER_PENALTY_GLARE = float(os.getenv("WEATHER_PENALTY_GLARE", "0.5"))
VISION_WEIGHT = float(os.getenv("VISION_WEIGHT", "0.6"))
PREDICTION_WEIGHT = float(os.getenv("PREDICTION_WEIGHT", "0.4"))

LOGGER = logging.getLogger(__name__)

model_lock = Lock()
workers_lock = Lock()
prediction_context_lock = Lock()
prediction_context = {}
camera_workers = {}
_loaded_model = None


def normalize_video_source(source_value):
    if isinstance(source_value, int):
        return source_value
    if source_value is None:
        return None
    source_text = str(source_value).strip()
    if source_text.isdigit():
        return int(source_text)
    return source_text


def extract_stream_from_html(html_value):
    if not html_value:
        return None
    text = str(html_value)
    stream_patterns = [
        r"https?://[^\s\"'<>]*CameraStream/[^\s\"'<>]+",
        r"https?://[^\s\"'<>]+\.m3u8[^\s\"'<>]*",
        r"src=[\"']([^\"']+)[\"']",
    ]

    for pattern in stream_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1) if match.lastindex else match.group(0)
        if candidate.startswith("//"):
            candidate = f"https:{candidate}"
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate

    return None


def load_camera_sources():
    camera_map = {}
    try:
        cameras = fetch_cameras()
        for location, data in cameras.items():
            stream_url = extract_stream_from_html(data.get("html"))
            if stream_url:
                camera_map[location] = normalize_video_source(stream_url)
    except Exception as exc:
        LOGGER.warning("Failed to load camera list: %s", exc)

    if DEFAULT_CAMERA_NAME not in camera_map:
        camera_map[DEFAULT_CAMERA_NAME] = normalize_video_source(DEFAULT_STREAM_SOURCE)

    return camera_map


camera_sources = load_camera_sources()
default_source = normalize_video_source(os.getenv("VIDEO_SOURCE", DEFAULT_STREAM_SOURCE))


def get_camera_raw_stream_url(camera_name):
    stream_source = camera_sources.get(camera_name)
    if isinstance(stream_source, str) and stream_source.startswith(("http://", "https://")):
        return stream_source
    return ""


def resolve_default_camera_name():
    if DEFAULT_CAMERA_NAME in camera_sources:
        return DEFAULT_CAMERA_NAME

    for camera_name, camera_source in camera_sources.items():
        if camera_source == default_source:
            return camera_name

    if camera_sources:
        return next(iter(camera_sources))

    return DEFAULT_CAMERA_NAME


default_camera_name = resolve_default_camera_name()


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value

def traffic_and_weather_rating(class_counts, coverage, predicted_vehicles, weather):
    """Calculates traffic score factoring in data from vision and predictive models
    that ranges from 0-10"""
    baseline_coverage = TRAFFIC_BASELINE_COVERAGE
    adjusted_coverage = max(coverage - baseline_coverage, 0)
    adjusted_max = max(100 - baseline_coverage, 1.0)

    car_count = class_counts.get("car", 0)
    motorcycle_count = class_counts.get("motorcycle", 0)
    bus_count = class_counts.get("bus", 0)
    truck_count = class_counts.get("truck", 0)
    total_count = car_count + motorcycle_count + bus_count + truck_count

    weighted_count = (car_count * 1.0) + (motorcycle_count * 0.5) + (bus_count * 2.5) + (truck_count * 3.0)
    weight_factor = weighted_count / max(total_count, 1.0)
    num_traffic_score = (adjusted_coverage / adjusted_max) * weight_factor
    num_traffic_score_0_to_10 = min(num_traffic_score * 10, 10)

    w_penalty = weather_penalty(weather)
    vision_score = num_traffic_score_0_to_10
    prediction_score = normalize_num_vehicles_prediction(predicted_vehicles)

    if prediction_score is not None:
        blended_score = (VISION_WEIGHT * vision_score) + (PREDICTION_WEIGHT * prediction_score)
    else:
        # If prediction_score has no data, vision score is fallback to avoid errors
        blended_score = vision_score

    # Calculating final score based on blended score (vision and predictive)
    # along with a weather penality, bounding it so it is no greater than 10
    composite = min(blended_score + w_penalty, 10)

    if composite <= TRAFFIC_LIGHT_MAX:
        text_traffic_score = "Light Traffic"
    elif composite <= TRAFFIC_MODERATE_MAX:
        text_traffic_score = "Moderate Traffic"
    elif composite <= TRAFFIC_HEAVY_MAX:
        text_traffic_score = "Heavy Traffic"
    else:
        text_traffic_score = "Very Heavy Traffic"

    return composite, text_traffic_score


def normalize_num_vehicles_prediction(predicted_num_vehicles):
    """Returns normalized number of vehicles predicted and returns number
    between 0 and 10"""
    if predicted_num_vehicles is None:
        return None

    return min((predicted_num_vehicles / PREDICTION_MAX_VEHICLES) * 10, 10)


def weather_penalty(weather):
    if weather is None:
        return 0.0
    penalty = 0.0

    if weather.get("is_snowing"):
        penalty += WEATHER_PENALTY_SNOW

    if weather.get("is_raining"):
        penalty += WEATHER_PENALTY_RAIN

    if weather.get("sun_glare"):
        penalty += WEATHER_PENALTY_GLARE

    return penalty

def vehicle_movement_rating(
    current_positions,
    previous_positions,
    stopped_threshold=MOVEMENT_STOPPED_THRESHOLD_PIXELS,
    slow_threshold=MOVEMENT_SLOW_THRESHOLD_PIXELS,
    max_match_distance=MOVEMENT_MAX_MATCH_DISTANCE_PIXELS,
):
    movement_counts = {"stopped": 0, "slow": 0, "fast": 0}
    if not previous_positions:
        return movement_counts

    available = list(previous_positions)

    for (cx, cy) in current_positions:
        best_dist = float("inf")
        best_idx = -1

        for i, (px, py) in enumerate(available):
            distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if distance < best_dist:
                best_dist = distance
                best_idx = i

        if best_idx < 0 or best_dist > max_match_distance:
            continue

        available.pop(best_idx)

        if best_dist < stopped_threshold:
            movement_counts["stopped"] += 1
        elif best_dist < slow_threshold:
            movement_counts["slow"] += 1
        else:
            movement_counts["fast"] += 1

    return movement_counts

def set_prediction_context(camera_name, predicted_vehicles, weather):
    """Sets element in prediction_context dictionary that stores camera name
    as key, and predicted vehicles and weather data as values"""

    # Lock is used to prevent multiple threads from accessing
    # prediction_context dictionary at once
    with prediction_context_lock:
        prediction_context[camera_name] = {
            "predicted_vehicles": predicted_vehicles,
            "weather": weather,
        }

def get_prediction_context(camera_name):
    """Gets predicted vehicles and weather data for camera name in
    prediction_context dictionary"""

    # Lock is used to prevent multiple threads from accessing
    # prediction_context dictionary at once
    with prediction_context_lock:
        context = prediction_context.get(camera_name)

        # If nothing is stored relating to a particular camera,
        # then return None
        if context is None:
            return None, None

        return context["predicted_vehicles"], context["weather"]

def get_empty_stats(camera_name):
    return {
        "vehicle_count": 0,
        "coverage": 0.0,
        "raw_coverage": 0.0,
        "traffic_score": 0.0,
        "traffic_label": "Light Traffic",
        "fps": 0.0,
        "resolution": None,
        "road_mask_percent": 0.0,
        "boxes_area": 0,
        "road_area": 0,
        "frame_area": 0,
        "road_learning_ready": False,
        "class_counts": {},
        "movement_counts": {"stopped": 0, "slow": 0, "fast": 0},
        "last_updated": "",
        "selected_camera": camera_name,
        "raw_stream_url": get_camera_raw_stream_url(camera_name),
    }


def get_yolo_model():
    global _loaded_model
    with model_lock:
        if _loaded_model is None:
            _loaded_model = YOLO(YOLO_MODEL)
        return _loaded_model


class CameraWorker:
    def __init__(self, camera_name, camera_source):
        self.camera_name = camera_name
        self.camera_source = camera_source
        self.lock = Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_stats = get_empty_stats(camera_name)
        self.active_viewers = 0
        self.last_accessed = time.monotonic()
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def touch(self):
        with self.lock:
            self.last_accessed = time.monotonic()

    def add_viewer(self):
        with self.lock:
            self.active_viewers += 1
            self.last_accessed = time.monotonic()

    def remove_viewer(self):
        with self.lock:
            self.active_viewers = max(0, self.active_viewers - 1)
            self.last_accessed = time.monotonic()

    def should_stop(self):
        with self.lock:
            idle_seconds = time.monotonic() - self.last_accessed
            return self.active_viewers == 0 and idle_seconds > WORKER_IDLE_TIMEOUT_SECONDS

    def run(self):
        cam = cv2.VideoCapture(self.camera_source)
        road_mask = None
        road_mask_last_seen = None
        smoothed_coverage = 0
        smoothed_fps = 0.0
        previous_frame_time = None
        first_mask_seen_time = None
        previous_positions = []
        generator_start_time = time.monotonic()

        while True:
            if self.should_stop():
                break

            if cam is None or not cam.isOpened():
                fallback = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    fallback,
                    f"Unable to open source: {self.camera_name}",
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    fallback,
                    str(self.camera_source)[:80],
                    (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                ok, buffer = cv2.imencode(".jpg", fallback)
                if ok:
                    with self.lock:
                        self.latest_frame = buffer.tobytes()
                        self.latest_frame_id += 1
                        self.latest_stats = get_empty_stats(self.camera_name)
                        self.latest_stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                time.sleep(0.5)
                cam = cv2.VideoCapture(self.camera_source)
                road_mask = None
                road_mask_last_seen = None
                smoothed_coverage = 0
                first_mask_seen_time = None
                generator_start_time = time.monotonic()
                continue

            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_width * frame_height
            frame_time = time.monotonic()
            if previous_frame_time is not None:
                delta_seconds = frame_time - previous_frame_time
                if delta_seconds > 0:
                    instant_fps = 1.0 / delta_seconds
                    smoothed_fps = smoothed_fps * (1.0 - FPS_SMOOTHING_ALPHA) + instant_fps * FPS_SMOOTHING_ALPHA
            previous_frame_time = frame_time

            model = get_yolo_model()
            with model_lock:
                results = model.predict(frame, conf=YOLO_CONFIDENCE, classes=YOLO_CLASS_IDS, verbose=False)

            if road_mask is None or road_mask.shape != (frame_height, frame_width):
                road_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                road_mask_last_seen = np.zeros((frame_height, frame_width), dtype=np.float64)
                first_mask_seen_time = None

            now = time.monotonic() - generator_start_time

            stale_mask = (road_mask > 0) & ((now - road_mask_last_seen) > ROAD_MASK_STALE_SECONDS)
            if np.any(stale_mask):
                faded_values = (road_mask[stale_mask].astype(np.float32) * ROAD_MASK_FADE_MULTIPLIER).astype(np.uint8)
                road_mask[stale_mask] = faded_values
                road_mask[road_mask < ROAD_MASK_MIN_VALUE] = 0
                road_mask_last_seen[road_mask == 0] = 0.0

            boxes_area = 0
            vehicle_count = 0
            class_counts = {}
            current_positions = []

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if y2 < frame_height * MIN_BOX_BOTTOM_Y_RATIO:
                        continue

                    vehicle_count += 1
                    class_name = model.names[cls]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    current_positions.append((cx, cy))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name.title()} {conf*100:.0f}%"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    boxes_area += (x2 - x1) * (y2 - y1)
                    road_mask[y1:y2, x1:x2] = np.maximum(road_mask[y1:y2, x1:x2], 200)
                    road_mask_last_seen[y1:y2, x1:x2] = now

            movement_counts = vehicle_movement_rating(current_positions, previous_positions)
            previous_positions = current_positions

            road_area = np.count_nonzero(road_mask)
            if road_area > 0 and first_mask_seen_time is None:
                first_mask_seen_time = now
            elif road_area == 0:
                first_mask_seen_time = None

            coverage_warmup_done = first_mask_seen_time is not None and (now - first_mask_seen_time) >= ROAD_MASK_WARMUP_SECONDS
            road_mask_percent = (road_area / frame_area) * 100 if frame_area > 0 else 0
            road_learning_ready = road_area > frame_area * ROAD_LEARNING_MIN_FRAME_RATIO
            raw_coverage = (boxes_area / road_area) * 100 if road_area > 0 and coverage_warmup_done else 0
            effective_coverage = raw_coverage if road_learning_ready else 0
            smoothed_coverage = smoothed_coverage * (1.0 - COVERAGE_SMOOTHING_ALPHA) + effective_coverage * COVERAGE_SMOOTHING_ALPHA
            coverage = smoothed_coverage

            predicted_vehicles, weather = get_prediction_context(self.camera_name)
            traffic_score, traffic_label = traffic_and_weather_rating(class_counts, coverage, predicted_vehicles, weather)

            mask_colored = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 1.0, mask_colored, 0.4, 0)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            stats_snapshot = {
                "vehicle_count": vehicle_count,
                "coverage": round(coverage, 2),
                "raw_coverage": round(raw_coverage, 2),
                "traffic_score": round(traffic_score, 2),
                "traffic_label": traffic_label,
                "fps": round(smoothed_fps, 2),
                "resolution": f"{frame_width}x{frame_height}",
                "road_mask_percent": round(road_mask_percent, 2),
                "boxes_area": int(boxes_area),
                "road_area": int(road_area),
                "frame_area": int(frame_area),
                "road_learning_ready": bool(road_learning_ready),
                "class_counts": class_counts,
                "movement_counts": movement_counts,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "selected_camera": self.camera_name,
                "raw_stream_url": get_camera_raw_stream_url(self.camera_name),
            }

            with self.lock:
                self.latest_frame = buffer.tobytes()
                self.latest_frame_id += 1
                self.latest_stats = stats_snapshot

        if cam is not None:
            cam.release()

        with workers_lock:
            if camera_workers.get(self.camera_name) is self:
                camera_workers.pop(self.camera_name, None)


def get_or_create_worker(camera_name):
    if camera_name not in camera_sources:
        camera_name = default_camera_name

    with workers_lock:
        worker = camera_workers.get(camera_name)
        if worker is None or not worker.thread.is_alive():
            worker = CameraWorker(camera_name, camera_sources[camera_name])
            camera_workers[camera_name] = worker

    worker.touch()
    return worker


def get_worker_snapshot(camera_name):
    worker = get_or_create_worker(camera_name)
    worker.touch()
    with worker.lock:
        frame_bytes = worker.latest_frame
        stats = dict(worker.latest_stats)
    return frame_bytes, to_jsonable(stats)


def list_camera_names():
    return list(camera_sources.keys())