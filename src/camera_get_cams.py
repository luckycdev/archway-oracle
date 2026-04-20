import requests
from config import CAMERAS_JSON_TIMEOUT_SECONDS, CAMERAS_JSON_URL


def fetch_cameras():
    response = requests.get(CAMERAS_JSON_URL, timeout=CAMERAS_JSON_TIMEOUT_SECONDS)
    response.raise_for_status()
    cams = response.json()

    camera_dict = {}
    for cam in cams:
        location = cam.get("location")
        if not location:
            continue
        camera_dict[location] = {
            "html": cam.get("html"),
            "x": cam.get("x"),
            "y": cam.get("y"),
        }

    return camera_dict