import logging
import math
import os
import re
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread
from urllib.parse import quote, unquote, urlparse

from camera_get_cams import fetch_cameras
from config import (
    CAMERA_BACKGROUND_DWELL_SECONDS,
    CAMERA_BACKGROUND_FRAME_STRIDE,
    CAMERA_BACKGROUND_SAMPLE_SECONDS,
    CAMERA_BACKGROUND_SCAN_ENABLED,
    CAMERA_BACKGROUND_WORKERS,
    CAMERA_BUFFER_SIZE,
    CAMERA_FRAME_DROP_GRABS,
    CAMERA_CAPTURE_OPEN_TIMEOUT_MS,
    CAMERA_CAPTURE_READ_TIMEOUT_MS,
    CAMERA_CAPTURE_RETRY_BACKOFF_SECONDS,
    CAMERA_SUPPRESS_OPENCV_LOGS,
    COVERAGE_SMOOTHING_ALPHA,
    DEFAULT_CAMERA_NAME,
    DEFAULT_STREAM_SOURCE,
    FPS_SMOOTHING_ALPHA,
    MIN_BOX_BOTTOM_Y_RATIO,
    MOVEMENT_MAX_MATCH_DISTANCE_PIXELS,
    MOVEMENT_SLOW_THRESHOLD_PIXELS,
    MOVEMENT_STOPPED_THRESHOLD_PIXELS,
    PREDICTION_MAX_VEHICLES,
    PREDICTION_WEIGHT,
    PROCESSED_STREAM_BIND_HOST,
    PROCESSED_STREAM_PORT,
    PROCESSED_STREAM_PUBLIC_HOST,
    ROAD_LEARNING_MIN_FRAME_RATIO,
    ROAD_MASK_FADE_MULTIPLIER,
    ROAD_MASK_MIN_VALUE,
    ROAD_MASK_STALE_SECONDS,
    ROAD_MASK_WARMUP_SECONDS,
    STREAM_JPEG_QUALITY,
    TRAFFIC_BASELINE_COVERAGE,
    TRAFFIC_HEAVY_MAX,
    TRAFFIC_LIGHT_MAX,
    TRAFFIC_MODERATE_MAX,
    VIDEO_SOURCE,
    VISION_WEIGHT,
    WEATHER_PENALTY_GLARE,
    WEATHER_PENALTY_RAIN,
    WEATHER_PENALTY_SNOW,
    WORKER_IDLE_TIMEOUT_SECONDS,
    YOLO_CLASS_IDS,
    YOLO_CONFIDENCE,
    YOLO_DEVICE,
    YOLO_DETECTION_INTERVAL_FRAMES,
    YOLO_MODEL,
    YOLO_MAX_VRAM_GB,
    YOLO_PROCESS_MAX_WIDTH,
)

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

model_lock = Lock()
workers_lock = Lock()
prediction_context_lock = Lock()
prediction_context = {}
camera_workers = {}
_loaded_model = None
_selected_device = None  # Store the selected YOLO device for reuse
processed_stream_lock = Lock()
processed_stream_server = None
processed_stream_thread = None
background_sampler_lock = Lock()
camera_status_lock = Lock()
background_queue_lock = Lock()
background_scan_state_lock = Lock()
background_stop_event = Event()
background_threads = []
background_queue = deque()
background_active_cameras = set()
camera_background_status = {}

STATUS_BADGE_LIGHT = "🟢"
STATUS_BADGE_MEDIUM = "🟡"
STATUS_BADGE_HEAVY = "🟠"
STATUS_BADGE_EXTREME = "🔴"
STATUS_BADGE_NO_FEED = "⚫"
STATUS_PREFIXES = (
    f"{STATUS_BADGE_LIGHT} ",
    f"{STATUS_BADGE_MEDIUM} ",
    f"{STATUS_BADGE_HEAVY} ",
    f"{STATUS_BADGE_EXTREME} ",
    f"{STATUS_BADGE_NO_FEED} ",
)
camera_open_retry_state = {}


def _silence_opencv_logging():
    if not CAMERA_SUPPRESS_OPENCV_LOGS:
        return

    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging") and hasattr(cv2.utils.logging, "LOG_LEVEL_SILENT"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
            return
    except Exception:
        pass

    try:
        if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_SILENT"):
            cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
            return
    except Exception:
        pass

    try:
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)
    except Exception:
        pass


def _redirect_stderr_to_null():
    try:
        stderr_fd = 2
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        os.close(devnull_fd)
    except Exception:
        pass


_silence_opencv_logging()
if CAMERA_SUPPRESS_OPENCV_LOGS:
    _redirect_stderr_to_null()


def _camera_open_retry_key(source_value):
    return str(source_value)


def _camera_open_is_allowed(source_value):
    retry_key = _camera_open_retry_key(source_value)
    next_retry_at = camera_open_retry_state.get(retry_key)
    return next_retry_at is None or time.monotonic() >= next_retry_at


def _mark_camera_open_failed(source_value):
    retry_key = _camera_open_retry_key(source_value)
    camera_open_retry_state[retry_key] = time.monotonic() + CAMERA_CAPTURE_RETRY_BACKOFF_SECONDS


def _mark_camera_open_success(source_value):
    retry_key = _camera_open_retry_key(source_value)
    camera_open_retry_state.pop(retry_key, None)


def open_video_capture(source_value):
    if not _camera_open_is_allowed(source_value):
        return None

    params = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CAMERA_CAPTURE_OPEN_TIMEOUT_MS])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, CAMERA_CAPTURE_READ_TIMEOUT_MS])

    try:
        if params:
            capture = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG, params)
            if capture is not None and capture.isOpened():
                _mark_camera_open_success(source_value)
                return capture
            _mark_camera_open_failed(source_value)
            return capture
    except Exception:
        _mark_camera_open_failed(source_value)
        pass

    try:
        capture = cv2.VideoCapture(source_value)
        if capture is not None and capture.isOpened():
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
            _mark_camera_open_success(source_value)
        else:
            _mark_camera_open_failed(source_value)
        return capture
    except Exception:
        _mark_camera_open_failed(source_value)
        return None


def select_and_test_yolo_device():
    """
    Select and test YOLO device with immediate console output.
    Tests the configured device and falls back to alternatives if needed.
    Returns the working device name.
    """
    configured_device = str(YOLO_DEVICE).strip() if YOLO_DEVICE is not None else "0"
    print(f"[YOLO Device Selection] Configured device: {configured_device}", flush=True)
    
    import torch
    
    # Build candidate devices to try
    candidates = []
    
    # Add configured device first
    if configured_device.lower() in {"cpu", "mps"}:
        candidates.append(configured_device.lower())
    else:
        try:
            candidates.append(str(int(configured_device)))  # Normalize GPU index
        except ValueError:
            candidates.append("0")  # Default to GPU 0
    
    # Add fallback options: GPU 0, GPU 1, CPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[YOLO Device Selection] CUDA available - found {gpu_count} GPU(s)", flush=True)
        for i in range(gpu_count):
            if str(i) not in candidates:
                candidates.append(str(i))
    else:
        print("[YOLO Device Selection] CUDA not available", flush=True)
    
    if "cpu" not in candidates:
        candidates.append("cpu")
    
    # Try each candidate
    selected_device = None
    for candidate_device in candidates:
        try:
            print(f"[YOLO Device Selection] Testing device: {candidate_device}...", end=" ", flush=True)
            
            # Try a simple tensor operation to validate device
            if candidate_device.lower() == "cpu":
                test_tensor = torch.zeros(1, device="cpu")
            elif candidate_device.lower() == "mps":
                test_tensor = torch.zeros(1, device="mps")
            else:
                # Try GPU device
                device_idx = int(candidate_device)
                if device_idx >= torch.cuda.device_count():
                    print(f"FAILED (GPU {device_idx} not available)", flush=True)
                    continue
                test_tensor = torch.zeros(1, device=f"cuda:{device_idx}")
                device_name = torch.cuda.get_device_name(device_idx)
                print(f"OK (found: {device_name})", flush=True)
            
            if test_tensor is not None:
                selected_device = candidate_device if candidate_device != "0" or torch.cuda.is_available() else candidate_device
                if candidate_device not in {"cpu", "mps"}:
                    selected_device = candidate_device
                else:
                    selected_device = candidate_device
                print(f"[YOLO Device Selection] ✓ Selected device: {selected_device}", flush=True)
                return selected_device
                
        except Exception as exc:
            print(f"FAILED ({type(exc).__name__}: {str(exc)[:50]})", flush=True)
            continue
    
    # Fallback to CPU if all else fails
    print("[YOLO Device Selection] ✓ All GPU devices failed, falling back to CPU", flush=True)
    return "cpu"


def get_effective_yolo_device():
    """Deprecated - use select_and_test_yolo_device() instead."""
    configured_device = str(YOLO_DEVICE).strip() if YOLO_DEVICE is not None else "0"
    if configured_device.lower() in {"cpu", "mps"}:
        return configured_device.lower()

    try:
        import torch

        if torch.cuda.is_available():
            return configured_device
    except Exception:
        pass

    LOGGER.warning("CUDA unavailable for YOLO_DEVICE=%s; falling back to CPU", configured_device)
    return "cpu"


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
default_source = normalize_video_source(VIDEO_SOURCE)


def reload_camera_sources():
    global camera_sources, default_camera_name

    refreshed_sources = load_camera_sources()

    camera_sources.clear()
    camera_sources.update(refreshed_sources)

    default_camera_name = resolve_default_camera_name()

    with background_queue_lock:
        background_queue.clear()
        background_queue.extend(camera_sources.keys())

    with background_scan_state_lock:
        background_active_cameras.clear()

    with camera_status_lock:
        current_names = set(camera_sources.keys())
        for camera_name in list(camera_background_status.keys()):
            if camera_name not in current_names:
                camera_background_status.pop(camera_name, None)

        for camera_name in current_names:
            camera_background_status.setdefault(camera_name, _get_default_status(camera_name))

    return list(camera_sources.keys())


def get_camera_raw_stream_url(camera_name):
    stream_source = camera_sources.get(camera_name)
    if isinstance(stream_source, str) and stream_source.startswith(("http://", "https://")):
        return stream_source
    return ""


def _badge_from_traffic_label(traffic_label):
    if traffic_label == "Light Traffic":
        return STATUS_BADGE_LIGHT
    if traffic_label == "Moderate Traffic":
        return STATUS_BADGE_MEDIUM
    if traffic_label == "Heavy Traffic":
        return STATUS_BADGE_HEAVY
    if traffic_label == "Very Heavy Traffic":
        return STATUS_BADGE_EXTREME
    return STATUS_BADGE_NO_FEED


def _get_default_status(camera_name):
    return {
        "camera_name": camera_name,
        "badge": STATUS_BADGE_NO_FEED,
        "traffic_label": "No Feed",
        "traffic_score": None,
        "vehicle_count": 0,
        "movement_counts": {"stopped": 0, "slow": 0, "fast": 0},
        "last_sampled": "",
    }


def _set_camera_background_status(camera_name, badge, traffic_label, traffic_score, stats=None):
    movement_counts = {"stopped": 0, "slow": 0, "fast": 0}
    vehicle_count = 0
    if isinstance(stats, dict):
        movement_counts = dict(stats.get("movement_counts", movement_counts) or movement_counts)
        vehicle_count = int(stats.get("vehicle_count", 0) or 0)

    with camera_status_lock:
        camera_background_status[camera_name] = {
            "camera_name": camera_name,
            "badge": badge,
            "traffic_label": traffic_label,
            "traffic_score": None if traffic_score is None else round(float(traffic_score), 2),
            "vehicle_count": vehicle_count,
            "movement_counts": movement_counts,
            "last_sampled": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def get_camera_background_status(camera_name):
    with camera_status_lock:
        return dict(camera_background_status.get(camera_name, _get_default_status(camera_name)))


def resolve_camera_name(camera_name_or_display):
    if camera_name_or_display in camera_sources:
        return camera_name_or_display

    if not isinstance(camera_name_or_display, str):
        return camera_name_or_display

    cleaned = camera_name_or_display.strip()
    for prefix in STATUS_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break

    return cleaned


def get_camera_display_name(camera_name):
    raw_name = resolve_camera_name(camera_name)
    if not CAMERA_BACKGROUND_SCAN_ENABLED:
        return raw_name

    status = get_camera_background_status(raw_name)
    badge = status.get("badge", STATUS_BADGE_NO_FEED)
    return f"{badge} {raw_name}"


def _sample_camera_traffic(camera_name, sample_seconds=CAMERA_BACKGROUND_SAMPLE_SECONDS):
    worker = get_or_create_worker(camera_name)
    started = time.monotonic()

    while time.monotonic() - started < sample_seconds:
        worker.touch()
        time.sleep(min(0.5, max(sample_seconds - (time.monotonic() - started), 0.1)))

    _, stats = get_worker_snapshot(camera_name)
    if not isinstance(stats, dict):
        return STATUS_BADGE_NO_FEED, "No Feed", None, None

    if not stats.get("last_updated") or stats.get("resolution") is None:
        return STATUS_BADGE_NO_FEED, "No Feed", None, stats

    traffic_score = stats.get("traffic_score")
    traffic_label = stats.get("traffic_label")
    if traffic_score is None or not traffic_label:
        return STATUS_BADGE_NO_FEED, "No Feed", None, stats

    return _badge_from_traffic_label(traffic_label), traffic_label, traffic_score, stats


def _background_sampler_loop():
    while not background_stop_event.is_set():
        camera_name = None

        with background_queue_lock:
            if not background_queue:
                background_queue.extend(camera_sources.keys())

            queue_length = len(background_queue)
            for _ in range(queue_length):
                candidate = background_queue.popleft()
                with background_scan_state_lock:
                    if candidate in background_active_cameras:
                        background_queue.append(candidate)
                        continue

                    background_active_cameras.add(candidate)
                    camera_name = candidate
                    break

        if not camera_name:
            time.sleep(0.2)
            continue

        try:
            badge, traffic_label, traffic_score, stats = _sample_camera_traffic(camera_name)
            _set_camera_background_status(camera_name, badge, traffic_label, traffic_score, stats=stats)
        except Exception as exc:
            LOGGER.warning("Background sample failed for %s: %s", camera_name, exc)
            _set_camera_background_status(camera_name, STATUS_BADGE_NO_FEED, "No Feed", None)
        finally:
            with background_scan_state_lock:
                background_active_cameras.discard(camera_name)

            with background_queue_lock:
                background_queue.append(camera_name)

        if CAMERA_BACKGROUND_DWELL_SECONDS > 0:
            time.sleep(CAMERA_BACKGROUND_DWELL_SECONDS)


def ensure_background_camera_sampler():
    if not CAMERA_BACKGROUND_SCAN_ENABLED:
        return False

    with background_sampler_lock:
        alive_threads = [thread for thread in background_threads if thread.is_alive()]
        if len(alive_threads) >= CAMERA_BACKGROUND_WORKERS:
            return True

        background_threads[:] = alive_threads
        if not background_queue:
            background_queue.extend(camera_sources.keys())

        with camera_status_lock:
            for camera_name in camera_sources:
                camera_background_status.setdefault(camera_name, _get_default_status(camera_name))

        while len(background_threads) < CAMERA_BACKGROUND_WORKERS:
            thread = Thread(target=_background_sampler_loop, daemon=True)
            thread.start()
            background_threads.append(thread)

    return True


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
    global _loaded_model, _selected_device
    with model_lock:
        if _loaded_model is None:
            try:
                import torch

                # Select and test device with console output (only happens once)
                _selected_device = select_and_test_yolo_device()
                print(f"[YOLO Model Init] Using device: {_selected_device}", flush=True)
                
                if _selected_device not in {"cpu", "mps"} and torch.cuda.is_available():
                    try:
                        device_index = int(str(_selected_device).strip())
                    except ValueError:
                        device_index = 0

                    total_vram_bytes = torch.cuda.get_device_properties(device_index).total_memory
                    total_vram_gb = total_vram_bytes / (1024 ** 3)
                    if total_vram_gb > 0:
                        memory_fraction = min(max(YOLO_MAX_VRAM_GB / total_vram_gb, 0.0), 0.99)
                        print(f"[YOLO Model Init] GPU {device_index}: {total_vram_gb:.2f}GB total, capping to {YOLO_MAX_VRAM_GB}GB ({memory_fraction*100:.1f}%)", flush=True)
                        torch.cuda.set_device(device_index)
                        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device_index)
            except Exception as exc:
                LOGGER.warning("Unable to apply YOLO CUDA memory cap: %s", exc)

            print(f"[YOLO Model Init] Loading model: {YOLO_MODEL}", flush=True)
            from ultralytics import YOLO

            _loaded_model = YOLO(YOLO_MODEL)
            print(f"[YOLO Model Init] Model loaded successfully", flush=True)
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
        cam = open_video_capture(self.camera_source)
        if cam is not None and cam.isOpened():
            cam.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        road_mask = None
        road_mask_last_seen = None
        smoothed_coverage = 0
        smoothed_fps = 0.0
        previous_frame_time = None
        first_mask_seen_time = None
        previous_positions = []
        generator_start_time = time.monotonic()
        frame_index = 0
        cached_detections = []
        latest_movement_counts = {"stopped": 0, "slow": 0, "fast": 0}
        model = get_yolo_model()

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
                cam = open_video_capture(self.camera_source)
                if cam is not None and cam.isOpened():
                    cam.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
                road_mask = None
                road_mask_last_seen = None
                smoothed_coverage = 0
                first_mask_seen_time = None
                generator_start_time = time.monotonic()
                continue

            for _ in range(CAMERA_FRAME_DROP_GRABS):
                cam.grab()

            ok, frame = cam.read()
            if not ok or frame is None:
                cam.release()
                cam = None
                _mark_camera_open_failed(self.camera_source)
                time.sleep(0.05)
                continue

            frame_height, frame_width = frame.shape[:2]
            frame_index += 1
            frame_area = frame_width * frame_height
            frame_time = time.monotonic()
            if previous_frame_time is not None:
                delta_seconds = frame_time - previous_frame_time
                if delta_seconds > 0:
                    instant_fps = 1.0 / delta_seconds
                    smoothed_fps = smoothed_fps * (1.0 - FPS_SMOOTHING_ALPHA) + instant_fps * FPS_SMOOTHING_ALPHA
            previous_frame_time = frame_time

            run_detection_this_frame = (frame_index % YOLO_DETECTION_INTERVAL_FRAMES) == 1
            if run_detection_this_frame:
                scale_x = 1.0
                scale_y = 1.0
                inference_frame = frame

                if YOLO_PROCESS_MAX_WIDTH > 0 and frame_width > YOLO_PROCESS_MAX_WIDTH:
                    resized_width = YOLO_PROCESS_MAX_WIDTH
                    resized_height = max(1, int(frame_height * (resized_width / frame_width)))
                    inference_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
                    scale_x = frame_width / float(resized_width)
                    scale_y = frame_height / float(resized_height)

                with model_lock:
                    results = model.predict(
                        inference_frame,
                        conf=YOLO_CONFIDENCE,
                        classes=YOLO_CLASS_IDS,
                        device=_selected_device,
                        verbose=False,
                    )

                refreshed_detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)

                        x1 = max(0, min(x1, frame_width - 1))
                        y1 = max(0, min(y1, frame_height - 1))
                        x2 = max(0, min(x2, frame_width))
                        y2 = max(0, min(y2, frame_height))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        refreshed_detections.append((x1, y1, x2, y2, conf, cls))

                cached_detections = refreshed_detections

            detections = cached_detections

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

            for (x1, y1, x2, y2, conf, cls) in detections:

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

            if run_detection_this_frame:
                latest_movement_counts = vehicle_movement_rating(current_positions, previous_positions)
                previous_positions = current_positions

            movement_counts = latest_movement_counts

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

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
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
    camera_name = resolve_camera_name(camera_name)
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
    camera_name = resolve_camera_name(camera_name)
    worker = get_or_create_worker(camera_name)
    worker.touch()
    with worker.lock:
        frame_bytes = worker.latest_frame
        stats = dict(worker.latest_stats)
    return frame_bytes, to_jsonable(stats)


def list_camera_names():
    return list(camera_sources.keys())


def _build_processed_stream_handler():
    class ProcessedStreamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"ok")
                return

            if not parsed.path.startswith("/processed/"):
                self.send_error(404, "Not found")
                return

            camera_name = unquote(parsed.path[len("/processed/") :])
            if not camera_name:
                self.send_error(400, "Missing camera name")
                return

            worker = get_or_create_worker(camera_name)
            worker.add_viewer()
            try:
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                last_frame_id = -1
                while True:
                    with worker.lock:
                        frame_bytes = worker.latest_frame
                        frame_id = worker.latest_frame_id

                    if not frame_bytes or frame_id == last_frame_id:
                        time.sleep(0.01)
                        continue

                    last_frame_id = frame_id
                    worker.touch()
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
            finally:
                worker.remove_viewer()

        def log_message(self, format, *args):
            return

    return ProcessedStreamHandler


def ensure_processed_stream_server():
    global processed_stream_server
    global processed_stream_thread

    with processed_stream_lock:
        if processed_stream_thread and processed_stream_thread.is_alive():
            return True

        try:
            processed_stream_server = ThreadingHTTPServer(
                (PROCESSED_STREAM_BIND_HOST, PROCESSED_STREAM_PORT),
                _build_processed_stream_handler(),
            )
        except OSError as exc:
            LOGGER.warning("Processed stream server failed to start on %s:%s (%s)", PROCESSED_STREAM_BIND_HOST, PROCESSED_STREAM_PORT, exc)
            processed_stream_server = None
            processed_stream_thread = None
            return False

        processed_stream_thread = Thread(target=processed_stream_server.serve_forever, daemon=True)
        processed_stream_thread.start()
        return True


def get_processed_stream_url(camera_name):
    if not camera_name:
        return ""

    if camera_name not in camera_sources:
        camera_name = default_camera_name

    if not ensure_processed_stream_server():
        return ""

    encoded_name = quote(camera_name, safe="")
    return f"/processed/{encoded_name}"