import math
import re

import plotly.graph_objects as go

from camera_get_cams import fetch_cameras


ROAD_STOP_WORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "boulevard", "blvd",
    "highway", "hwy", "route", "rt", "interstate", "i", "us", "north",
    "south", "east", "west", "n", "s", "e", "w",
}


def load_camera_points(allowed_locations=None):
    allowed = set(allowed_locations) if allowed_locations else None
    camera_points = []

    cameras = fetch_cameras()
    for location, data in cameras.items():
        if allowed is not None and location not in allowed:
            continue

        x = data.get("x")
        y = data.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            camera_points.append({"location": location, "x": x, "y": y})

    return camera_points


def haversine_miles(lat1, lon1, lat2, lon2):
    to_rad = math.pi / 180
    d_lat = (lat2 - lat1) * to_rad
    d_lon = (lon2 - lon1) * to_rad
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1 * to_rad) * math.cos(lat2 * to_rad) * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    earth_radius_miles = 3958.8
    return earth_radius_miles * c


def get_compass_direction(lat1, lon1, lat2, lon2):
    to_rad = math.pi / 180
    to_deg = 180 / math.pi
    lat1_rad = lat1 * to_rad
    lat2_rad = lat2 * to_rad
    d_lon = (lon2 - lon1) * to_rad

    y = math.sin(d_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lon)
    bearing = (math.atan2(y, x) * to_deg + 360) % 360

    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    direction_index = round(bearing / 45) % len(directions)
    return directions[direction_index]


def get_nearby_cameras(camera_points, current_camera, limit=5):
    if not current_camera:
        return []

    map_camera_by_name = {point["location"]: point for point in camera_points if "location" in point}
    origin = map_camera_by_name.get(current_camera)
    if not origin:
        return []

    nearest = []
    for camera in camera_points:
        if camera.get("location") == current_camera:
            continue

        miles = haversine_miles(origin["y"], origin["x"], camera["y"], camera["x"])
        direction = get_compass_direction(origin["y"], origin["x"], camera["y"], camera["x"])
        nearest.append({"camera": camera, "miles": miles, "direction": direction})

    nearest.sort(key=lambda item: item["miles"])
    return nearest[:limit]


def _normalize_tokens(text):
    if not text:
        return set()
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    tokens = {token for token in raw_tokens if token and token not in ROAD_STOP_WORDS and len(token) > 1}
    return tokens


def get_cameras_along_road(camera_points, road_name, limit=8):
    road_tokens = _normalize_tokens(road_name)
    if not road_tokens:
        return []

    matches = []
    for point in camera_points:
        location = point.get("location", "")
        location_tokens = _normalize_tokens(location)
        shared = road_tokens.intersection(location_tokens)
        if shared:
            matches.append({"camera": point, "shared_tokens": sorted(shared)})

    matches.sort(key=lambda item: (-len(item["shared_tokens"]), item["camera"].get("location", "")))
    return matches[:limit]


def get_cameras_near_road(camera_points, road_latitude, road_longitude, limit=8):
    if road_latitude is None or road_longitude is None:
        return []

    near = []
    for point in camera_points:
        miles = haversine_miles(road_latitude, road_longitude, point["y"], point["x"])
        direction = get_compass_direction(road_latitude, road_longitude, point["y"], point["x"])
        near.append({"camera": point, "miles": miles, "direction": direction})

    near.sort(key=lambda item: item["miles"])
    return near[:limit]


def build_camera_map_figure(camera_points, selected_camera=None):
    if not camera_points:
        return go.Figure()

    lats = []
    lons = []
    names = []
    colors = []
    sizes = []

    for point in camera_points:
        lats.append(point["y"])
        lons.append(point["x"])
        names.append(point["location"])
        is_selected = point["location"] == selected_camera
        colors.append("#ef4444" if is_selected else "#22c55e")
        sizes.append(17 if is_selected else 10)

    fig = go.Figure(
        go.Scattermap(
            lat=lats,
            lon=lons,
            mode="markers",
            marker={"size": sizes, "color": colors, "opacity": 0.95},
            text=names,
            customdata=names,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        map={
            "style": "open-street-map",
            "center": {"lat": 38.5733, "lon": -92.6041},
            "zoom": 7,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        clickmode="event+select",
        height=460,
        uirevision="camera-map-fixed-ui",
    )
    return fig


def extract_selected_camera_from_map_event(selection):
    if not selection:
        return None

    selected_points = selection.get("selection", {}).get("points", [])
    if not selected_points:
        return None

    selected_point = selected_points[0]
    custom_data = selected_point.get("customdata")
    if isinstance(custom_data, str):
        return custom_data

    if isinstance(custom_data, (list, tuple)) and custom_data:
        return str(custom_data[0])

    return None