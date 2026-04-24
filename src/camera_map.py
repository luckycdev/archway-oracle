import math
import re
from collections.abc import Mapping

import plotly.graph_objects as go

from camera_get_cams import fetch_cameras
from config import (
    CAMERA_TRIANGLE_A_LAT,
    CAMERA_TRIANGLE_A_LON,
    CAMERA_TRIANGLE_B_LAT,
    CAMERA_TRIANGLE_B_LON,
    CAMERA_TRIANGLE_C_LAT,
    CAMERA_TRIANGLE_C_LON,
    CAMERA_TRIANGLE_FILTER_ENABLED,
)


ROAD_STOP_WORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "boulevard", "blvd",
    "highway", "hwy", "route", "rt", "interstate", "i", "us", "north",
    "south", "east", "west", "n", "s", "e", "w",
}


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
            if CAMERA_TRIANGLE_FILTER_ENABLED and not _is_point_in_triangle(y, x):
                continue
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


def _map_color_from_status(status, background_scan_enabled):
    if not background_scan_enabled:
        return "#22c55e"

    if not status:
        return "#000000"

    badge = status.get("badge")
    if badge == "🟢":
        return "#22c55e"
    if badge == "🟡":
        return "#eab308"
    if badge == "🟠":
        return "#f97316"
    if badge == "🔴":
        return "#ef4444"
    return "#000000"


def build_camera_map_figure(camera_points, selected_camera=None, camera_statuses=None, background_scan_enabled=True):
    if not camera_points:
        return go.Figure()

    camera_statuses = camera_statuses or {}

    base_lats = []
    base_lons = []
    base_names = []
    base_colors = []
    base_ids = []

    selected_lats = []
    selected_lons = []
    selected_names = []
    selected_colors = []
    selected_ids = []

    for point in camera_points:
        is_selected = point["location"] == selected_camera
        color = _map_color_from_status(camera_statuses.get(point["location"]), background_scan_enabled)
        if is_selected:
            selected_lats.append(point["y"])
            selected_lons.append(point["x"])
            selected_names.append(point["location"])
            selected_colors.append("#a855f7")
            selected_ids.append(point["location"])
        else:
            base_lats.append(point["y"])
            base_lons.append(point["x"])
            base_names.append(point["location"])
            base_colors.append(color)
            base_ids.append(point["location"])

    fig = go.Figure()

    if base_lats:
        fig.add_trace(
            go.Scattermapbox(
                lat=base_lats,
                lon=base_lons,
                mode="markers",
                marker={"size": 10, "color": base_colors, "opacity": 0.95},
                text=base_names,
                customdata=base_names,
                ids=base_ids,
                showlegend=False,
                hovertemplate="%{text}<extra></extra>",
                hoverinfo="skip",
            )
        )

    if selected_lats:
        fig.add_trace(
            go.Scattermapbox(
                lat=selected_lats,
                lon=selected_lons,
                mode="markers",
                marker={"size": 22, "color": selected_colors, "opacity": 1.0},
                text=selected_names,
                customdata=selected_names,
                ids=selected_ids,
                showlegend=False,
                hovertemplate="%{text}<extra></extra>",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        mapbox={
            "style": "open-street-map",
            "center": {"lat": 38.5733, "lon": -92.6041},
            "zoom": 7,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        clickmode="event+select",
        showlegend=False,
        height=460,
        uirevision="camera-map-fixed-ui",
    )
    return fig


def _get_event_value(obj, key, default=None):
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_points_from_selection(selection):
    if not selection:
        return []

    if isinstance(selection, Mapping):
        nested_selection = selection.get("selection")
        if isinstance(nested_selection, Mapping):
            points = nested_selection.get("points")
            if points:
                return points

        points = selection.get("points")
        if points:
            return points

        if isinstance(nested_selection, list):
            return nested_selection

        return []

    nested_selection = getattr(selection, "selection", None)
    if nested_selection is not None:
        points = getattr(nested_selection, "points", None)
        if points:
            return points

    points = getattr(selection, "points", None)
    if points:
        return points

    return []


def _resolve_camera_from_point_point(selected_point, camera_points=None):
    custom_data = _get_event_value(selected_point, "customdata")
    if isinstance(custom_data, str):
        return custom_data

    if isinstance(custom_data, (list, tuple)) and custom_data:
        return str(custom_data[0])

    text_value = _get_event_value(selected_point, "text")
    if isinstance(text_value, str) and text_value:
        return text_value

    if not camera_points:
        return None

    lat = _get_event_value(selected_point, "lat")
    lon = _get_event_value(selected_point, "lon")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None

    closest_camera = None
    closest_distance = None
    for camera in camera_points:
        camera_lat = camera.get("y")
        camera_lon = camera.get("x")
        if not isinstance(camera_lat, (int, float)) or not isinstance(camera_lon, (int, float)):
            continue

        distance = (camera_lat - lat) ** 2 + (camera_lon - lon) ** 2
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_camera = camera.get("location")

    return closest_camera


def extract_selected_camera_from_map_event(selection, camera_points=None):
    selected_points = _extract_points_from_selection(selection)

    if not selected_points:
        return None

    for selected_point in selected_points:
        selected_camera = _resolve_camera_from_point_point(selected_point, camera_points)
        if selected_camera:
            return selected_camera

    selected_point = selected_points[0]
    return _resolve_camera_from_point_point(selected_point, camera_points)