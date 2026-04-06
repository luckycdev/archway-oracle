import os

import requests
from dotenv import load_dotenv


load_dotenv()

CAMERAS_JSON_URL = os.getenv(
    "CAMERAS_JSON_URL",
    "https://traveler.modot.org/timconfig/feed/desktop/StreamingCams2.json",
)


def fetch_cameras():
    response = requests.get(CAMERAS_JSON_URL, timeout=15)
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