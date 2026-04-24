import os
from datetime import datetime
from typing import List

from dotenv import load_dotenv


load_dotenv()
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "loglevel;quiet|rtsp_transport;tcp")


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_int_list(name: str, default: List[int]) -> List[int]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    items = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            items.append(int(chunk))
        except ValueError:
            continue

    return items if items else default


# App UI
APP_PAGE_TITLE = _get_str("APP_PAGE_TITLE", "Archway Oracle: St. Louis Traffic Predictive Intelligence & Detector")
APP_PAGE_ICON = _get_str("APP_PAGE_ICON", "🚗")
APP_PAGE_LAYOUT = _get_str("APP_PAGE_LAYOUT", "wide")
LIVE_STATS_REFRESH_INTERVAL = _get_str("LIVE_STATS_REFRESH_INTERVAL", "450ms")
CAMERA_LIVE_UPDATE_DEFAULT = _get_bool("CAMERA_LIVE_UPDATE_DEFAULT", True)

# Commute recommendation tuning
PATIENCE_MIN = _get_int("PATIENCE_MIN", 100)
PATIENCE_MAX = _get_int("PATIENCE_MAX", 1000)
PATIENCE_DEFAULT = _get_int("PATIENCE_DEFAULT", 600)
PATIENCE_STEP = _get_int("PATIENCE_STEP", 50)
LOOKAHEAD_HOURS = _get_int("LOOKAHEAD_HOURS", 4)
WEATHER_SCORE_PENALTY_SNOW = _get_float("WEATHER_SCORE_PENALTY_SNOW", 1000.0)
WEATHER_SCORE_PENALTY_RAIN = _get_float("WEATHER_SCORE_PENALTY_RAIN", 400.0)
WEATHER_SCORE_PENALTY_GLARE = _get_float("WEATHER_SCORE_PENALTY_GLARE", 300.0)
COMMUTE_MINUTES_PER_100_VEHICLES = _get_float("COMMUTE_MINUTES_PER_100_VEHICLES", 2.0)
COMMUTE_GALLONS_PER_5_MIN = _get_float("COMMUTE_GALLONS_PER_5_MIN", 0.05)
GAS_PRICE_PER_GALLON = _get_float("GAS_PRICE_PER_GALLON", 3.25)

# Data processing
DATA_FILE_PATH = _get_str("DATA_FILE_PATH", "stl_traffic_counts.csv")
DATA_MIN_YEAR = _get_int("DATA_MIN_YEAR", 2023)
GPS_LAT_BASE = _get_float("GPS_LAT_BASE", 35.8134)
GPS_LAT_SCALE = _get_float("GPS_LAT_SCALE", 0.000002778)
GPS_LON_BASE = _get_float("GPS_LON_BASE", -93.3488)
GPS_LON_SCALE = _get_float("GPS_LON_SCALE", 0.000003465)

SYNTH_DAYS_BACK = _get_int("SYNTH_DAYS_BACK", 1)
SYNTH_DAYS_FORWARD = _get_int("SYNTH_DAYS_FORWARD", 1)
SYNTH_NOISE_MIN = _get_float("SYNTH_NOISE_MIN", 0.85)
SYNTH_NOISE_MAX = _get_float("SYNTH_NOISE_MAX", 1.15)

MORNING_WEATHER_START_HOUR = _get_int("MORNING_WEATHER_START_HOUR", 6)
MORNING_WEATHER_END_HOUR = _get_int("MORNING_WEATHER_END_HOUR", 10)
MORNING_SIM_PRECIPITATION = _get_float("MORNING_SIM_PRECIPITATION", 4.5)
MORNING_SIM_TEMPERATURE = _get_float("MORNING_SIM_TEMPERATURE", -2.0)

OPEN_METEO_BASE_URL = _get_str("OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1/forecast")
OPEN_METEO_LATITUDE = _get_float("OPEN_METEO_LATITUDE", 38.6274)
OPEN_METEO_LONGITUDE = _get_float("OPEN_METEO_LONGITUDE", -90.1982)
OPEN_METEO_TIMEOUT_SECONDS = _get_float("OPEN_METEO_TIMEOUT_SECONDS", 5.0)
WEATHER_FALLBACK_TEMPERATURE = _get_float("WEATHER_FALLBACK_TEMPERATURE", 15.0)
WEATHER_FALLBACK_PRECIPITATION = _get_float("WEATHER_FALLBACK_PRECIPITATION", 0.0)
WEATHER_FALLBACK_CLOUD_COVER = _get_float("WEATHER_FALLBACK_CLOUD_COVER", 0.0)

SUN_GLARE_CLOUD_THRESHOLD = _get_float("SUN_GLARE_CLOUD_THRESHOLD", 30.0)
SUN_GLARE_HOURS = _get_int_list("SUN_GLARE_HOURS", [7, 8, 17, 18])
HOLIDAY_YEARS = _get_int_list("HOLIDAY_YEARS", [datetime.now().year])

# Model training
MODEL_RANDOM_STATE = _get_int("MODEL_RANDOM_STATE", 42)
MODEL_TEST_SPLIT_RATIO = _get_float("MODEL_TEST_SPLIT_RATIO", 0.8)
MODEL_CV_SPLITS = _get_int("MODEL_CV_SPLITS", 3)
MODEL_SEARCH_ITERATIONS = _get_int("MODEL_SEARCH_ITERATIONS", 10)
MODEL_PERMUTATION_REPEATS = _get_int("MODEL_PERMUTATION_REPEATS", 3)
MODEL_SEARCH_N_JOBS = _get_int("MODEL_SEARCH_N_JOBS", -1)
MODEL_SEARCH_VERBOSE = _get_int("MODEL_SEARCH_VERBOSE", 1)

# Camera feed catalog
CAMERAS_JSON_URL = _get_str(
    "CAMERAS_JSON_URL",
    "https://traveler.modot.org/timconfig/feed/desktop/StreamingCams2.json",
)
CAMERAS_JSON_TIMEOUT_SECONDS = _get_float("CAMERAS_JSON_TIMEOUT_SECONDS", 15.0)
CAMERA_TRIANGLE_FILTER_ENABLED = _get_bool("CAMERA_TRIANGLE_FILTER_ENABLED", True)

# Triangle vertices (lat, lon) to keep cameras around St. Louis metro only.
CAMERA_TRIANGLE_A_LAT = _get_float("CAMERA_TRIANGLE_A_LAT", 39.10)
CAMERA_TRIANGLE_A_LON = _get_float("CAMERA_TRIANGLE_A_LON", -90.95)
CAMERA_TRIANGLE_B_LAT = _get_float("CAMERA_TRIANGLE_B_LAT", 38.05)
CAMERA_TRIANGLE_B_LON = _get_float("CAMERA_TRIANGLE_B_LON", -91.35)
CAMERA_TRIANGLE_C_LAT = _get_float("CAMERA_TRIANGLE_C_LAT", 38.00)
CAMERA_TRIANGLE_C_LON = _get_float("CAMERA_TRIANGLE_C_LON", -89.75)

# Camera worker / CV pipeline
SYMPY_GROUND_TYPES = _get_str("SYMPY_GROUND_TYPES", "python")
os.environ.setdefault("SYMPY_GROUND_TYPES", SYMPY_GROUND_TYPES)

YOLO_MODEL = _get_str("YOLO_MODEL", "yolo26n.pt")
YOLO_DEVICE = _get_str("YOLO_DEVICE", "0")
YOLO_MAX_VRAM_GB = _get_float("YOLO_MAX_VRAM_GB", 5.5)
YOLO_CONFIDENCE = _get_float("YOLO_CONFIDENCE", 0.15)
YOLO_CLASS_IDS = _get_int_list("YOLO_CLASS_IDS", [2, 3, 5, 7])
MIN_BOX_BOTTOM_Y_RATIO = _get_float("MIN_BOX_BOTTOM_Y_RATIO", 0.3)
WORKER_IDLE_TIMEOUT_SECONDS = _get_int("WORKER_IDLE_TIMEOUT_SECONDS", 10)
MOVEMENT_STOPPED_THRESHOLD_PIXELS = _get_float("MOVEMENT_STOPPED_THRESHOLD_PIXELS", 0.5)
MOVEMENT_SLOW_THRESHOLD_PIXELS = _get_float("MOVEMENT_SLOW_THRESHOLD_PIXELS", 4.0)
MOVEMENT_MAX_MATCH_DISTANCE_PIXELS = _get_float("MOVEMENT_MAX_MATCH_DISTANCE_PIXELS", 100.0)
ROAD_MASK_STALE_SECONDS = _get_float("ROAD_MASK_STALE_SECONDS", 5.0)
ROAD_MASK_WARMUP_SECONDS = _get_float("ROAD_MASK_WARMUP_SECONDS", 0.5)
ROAD_MASK_FADE_MULTIPLIER = _get_float("ROAD_MASK_FADE_MULTIPLIER", 0.95)
ROAD_MASK_MIN_VALUE = _get_int("ROAD_MASK_MIN_VALUE", 5)
ROAD_LEARNING_MIN_FRAME_RATIO = _get_float("ROAD_LEARNING_MIN_FRAME_RATIO", 0.05)
FPS_SMOOTHING_ALPHA = _get_float("FPS_SMOOTHING_ALPHA", 0.2)
COVERAGE_SMOOTHING_ALPHA = _get_float("COVERAGE_SMOOTHING_ALPHA", 0.2)
TRAFFIC_BASELINE_COVERAGE = _get_float("TRAFFIC_BASELINE_COVERAGE", 6.5)
TRAFFIC_LIGHT_MAX = _get_float("TRAFFIC_LIGHT_MAX", 2.5)
TRAFFIC_MODERATE_MAX = _get_float("TRAFFIC_MODERATE_MAX", 5.0)
TRAFFIC_HEAVY_MAX = _get_float("TRAFFIC_HEAVY_MAX", 7.5)
PREDICTION_MAX_VEHICLES = _get_float("PREDICTION_MAX_VEHICLES", 2500.0)
WEATHER_PENALTY_SNOW = _get_float("WEATHER_PENALTY_SNOW", 1.5)
WEATHER_PENALTY_RAIN = _get_float("WEATHER_PENALTY_RAIN", 0.8)
WEATHER_PENALTY_GLARE = _get_float("WEATHER_PENALTY_GLARE", 0.5)
VISION_WEIGHT = _get_float("VISION_WEIGHT", 0.6)
PREDICTION_WEIGHT = _get_float("PREDICTION_WEIGHT", 0.4)
YOLO_PROCESS_MAX_WIDTH = _get_int("YOLO_PROCESS_MAX_WIDTH", 640)
YOLO_DETECTION_INTERVAL_FRAMES = max(1, _get_int("YOLO_DETECTION_INTERVAL_FRAMES", 2))
YOLO_DETECTION_MIN_INTERVAL_SECONDS = max(0.0, _get_float("YOLO_DETECTION_MIN_INTERVAL_SECONDS", 0.0))
CAMERA_BUFFER_SIZE = max(1, _get_int("CAMERA_BUFFER_SIZE", 1))
CAMERA_FRAME_DROP_GRABS = max(0, _get_int("CAMERA_FRAME_DROP_GRABS", 1))
STREAM_JPEG_QUALITY = max(40, min(100, _get_int("STREAM_JPEG_QUALITY", 75)))
STREAM_PLAYBACK_DELAY_SECONDS = max(0.0, _get_float("STREAM_PLAYBACK_DELAY_SECONDS", 5.0))
STREAM_OUTPUT_FPS = max(1.0, _get_float("STREAM_OUTPUT_FPS", 10.0))
STREAM_HISTORY_MAX_FRAMES = max(30, _get_int("STREAM_HISTORY_MAX_FRAMES", 400))
PROCESSED_STREAM_BIND_HOST = _get_str("PROCESSED_STREAM_BIND_HOST", "127.0.0.1")
PROCESSED_STREAM_PUBLIC_HOST = _get_str("PROCESSED_STREAM_PUBLIC_HOST", "127.0.0.1")
PROCESSED_STREAM_PORT = _get_int("PROCESSED_STREAM_PORT", 8765)

# Background camera sampling queue
CAMERA_BACKGROUND_SCAN_ENABLED = _get_bool("CAMERA_BACKGROUND_SCAN_ENABLED", True)
CAMERA_BACKGROUND_SAMPLE_SECONDS = max(1.0, _get_float("CAMERA_BACKGROUND_SAMPLE_SECONDS", 5.0))
CAMERA_BACKGROUND_DWELL_SECONDS = max(0.0, _get_float("CAMERA_BACKGROUND_DWELL_SECONDS", 0.25))
CAMERA_BACKGROUND_WORKERS = max(1, _get_int("CAMERA_BACKGROUND_WORKERS", 1))
CAMERA_BACKGROUND_FRAME_STRIDE = max(1, _get_int("CAMERA_BACKGROUND_FRAME_STRIDE", 2))
CAMERA_CAPTURE_OPEN_TIMEOUT_MS = max(1000, _get_int("CAMERA_CAPTURE_OPEN_TIMEOUT_MS", 5000))
CAMERA_CAPTURE_READ_TIMEOUT_MS = max(1000, _get_int("CAMERA_CAPTURE_READ_TIMEOUT_MS", 5000))
CAMERA_CAPTURE_RETRY_BACKOFF_SECONDS = max(5.0, _get_float("CAMERA_CAPTURE_RETRY_BACKOFF_SECONDS", 60.0))
CAMERA_SUPPRESS_OPENCV_LOGS = _get_bool("CAMERA_SUPPRESS_OPENCV_LOGS", True)
