import requests
from config import (
    CAMERA_TRIANGLE_A_LAT,
    CAMERA_TRIANGLE_A_LON,
    CAMERA_TRIANGLE_B_LAT,
    CAMERA_TRIANGLE_B_LON,
    CAMERA_TRIANGLE_C_LAT,
    CAMERA_TRIANGLE_C_LON,
    CAMERA_TRIANGLE_FILTER_ENABLED,
    CAMERAS_JSON_TIMEOUT_SECONDS,
    CAMERAS_JSON_URL,
)


def _point_side(px, py, ax, ay, bx, by):
    return (px - bx) * (ay - by) - (ax - bx) * (py - by)


def _is_point_in_triangle(lat, lon):
    ax, ay = CAMERA_TRIANGLE_A_LAT, CAMERA_TRIANGLE_A_LON
    bx, by = CAMERA_TRIANGLE_B_LAT, CAMERA_TRIANGLE_B_LON
    cx, cy = CAMERA_TRIANGLE_C_LAT, CAMERA_TRIANGLE_C_LON

    d1 = _point_side(lat, lon, ax, ay, bx, by)
    d2 = _point_side(lat, lon, bx, by, cx, cy)
    d3 = _point_side(lat, lon, cx, cy, ax, ay)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def fetch_cameras():
    response = requests.get(CAMERAS_JSON_URL, timeout=CAMERAS_JSON_TIMEOUT_SECONDS)
    response.raise_for_status()
    cams = response.json()

    camera_dict = {}
    for cam in cams:
        location = cam.get("location")
        if not location:
            continue

        x = cam.get("x")
        y = cam.get("y")
        if CAMERA_TRIANGLE_FILTER_ENABLED:
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            if not _is_point_in_triangle(y, x):
                continue

        camera_dict[location] = {
            "html": cam.get("html"),
            "x": x,
            "y": y,
        }

    return camera_dict