import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from html import escape

# Import from our local src modules
from data_processing import load_and_prep_data, classify_traffic
from model import train_and_evaluate
from visualizations import build_traffic_chart, build_feature_importance_chart # Added this
from engine import get_best_time_to_leave
from camera_map import (
    build_camera_map_figure,
    extract_selected_camera_from_map_event,
    get_cameras_along_road,
    get_cameras_near_road,
    load_camera_points,
)
from camera_ui import render_camera_stats
from camera_workers import get_processed_stream_url, get_worker_snapshot, list_camera_names, set_prediction_context

# --- 1. Page Configuration ---
st.set_page_config(page_title="ArchWay Oracle: St. Louis Traffic Predictive Intelligence & Detector", layout="wide", page_icon="🚗")
st.title("🚗 ArchWay Oracle: St. Louis Traffic Predictive Intelligence & Detector")

# --- 2. Load Data & Model ---
@st.cache_resource
def get_data_and_model():
    df = load_and_prep_data()
    # Added feature_importances to the return
    test_data, ai_mae, baseline_mae, cv_scores, winning_params, feature_importances = train_and_evaluate(df)
    return df, test_data, ai_mae, baseline_mae, winning_params, feature_importances

with st.spinner("Analyzing St. Louis Road Network..."):
    full_data, test_results, ai_mae, baseline_mae, winning_params, feature_importances = get_data_and_model()

# --- 3. Sidebar Controls ---
st.sidebar.header("🕹️ Control Panel")


# --- Select Primary Roadway (controlled by session state) ---
all_segments = sorted(test_results['road_segment_id'].unique())
if 'selected_segment' not in st.session_state:
    st.session_state['selected_segment'] = all_segments[0]
# If a camera or button set a new segment, update before widget is created
if '_pending_selected_segment' in st.session_state:
    st.session_state['selected_segment'] = st.session_state.pop('_pending_selected_segment')
selected_segment = st.sidebar.selectbox(
    "1. Select Primary Roadway",
    all_segments,
    index=all_segments.index(st.session_state['selected_segment']) if st.session_state['selected_segment'] in all_segments else 0,
    key='selected_segment',
)

# Route Comparison Toggle
compare_on = st.sidebar.checkbox("🔗 Enable Route Comparison")
secondary_segment = None
if compare_on:
    secondary_segment = st.sidebar.selectbox(
        "Select Secondary Roadway", 
        [s for s in all_segments if s != selected_segment]
    )

st.sidebar.markdown("---")

st.sidebar.subheader("📅 Temporal Settings")

# Limits the calendar to the 3 days of data we generated
min_date = datetime.now().date() - timedelta(days=1)
max_date = datetime.now().date() + timedelta(days=1)

selected_day = st.sidebar.date_input(
    "Select Date",
    value=datetime.now().date(),
    min_value=min_date,
    max_value=max_date
)

selected_hour = st.sidebar.slider(
    "Set Prediction Hour",
    min_value=0,
    max_value=23,
    value=datetime.now().hour,
    format="%d:00"
)

# Combine Day and Hour into one DateTime object for filtering
selected_date = datetime.combine(selected_day, datetime.min.time()) + timedelta(hours=selected_hour)

st.sidebar.info(f"📍 Viewing: **{selected_date.strftime('%a, %b %d at %I:%M %p')}**")

# Patience slider
st.sidebar.subheader("🧠 AI Persona")
# Add a slider to let the user define how much they hate waiting
patience = st.sidebar.slider(
    "Commuter Patience", 
    min_value=100, 
    max_value=1000, 
    value=600, 
    step=50,
    help="Higher = Wants to leave ASAP. Lower = Willing to wait for clear roads."
)

# --- 4. Logic: Data Filtering ---
segment_df = test_results[test_results['road_segment_id'] == selected_segment].sort_values('DateTime')
history_data = segment_df[segment_df['DateTime'] <= selected_date]
future_data = segment_df[segment_df['DateTime'] > selected_date].copy()

# --- 5. Key Metrics Display & AI Tip ---
st.markdown(f"### 📍 Analysis: {selected_segment}")
current_row = segment_df[segment_df['DateTime'] == selected_date]

if not current_row.empty:
    row = current_row.iloc[0]
    m1, m2, m3, m4 = st.columns(4)

    if st.session_state.get("selected_camera"):

        # Building weather dictionary
        weather= {
            "is_snowing": int(row.get("is_snowing", 0)),
            "is_raining": int(row.get("is_raining", 0)),
            "sun_glare": int(row.get("sun_glare", 0)),
        }

        # Calling set_prediction_context function to wrap weather data into
        # internal dictionary
        set_prediction_context(
            camera_name = st.session_state.selected_camera,
            predicted_vehicles = float(row["Predicted_Vehicles"]),
            weather = weather
        )
    
    # 1. Traffic Volume
    m1.metric("Current Vol", f"{int(row['vehicle_count'])}/hr")
    
    # 2. AI Prediction with Delta (Shows how much the AI differs from reality)
    ai_diff = int(row['Predicted_Vehicles'] - row['vehicle_count'])
    m2.metric("AI Forecast", f"{int(row['Predicted_Vehicles'])}/hr", delta=ai_diff, delta_color="off")
    
    # 3. Dynamic Weather Metric
    temp = row.get('temperature', 20)
    precip = row.get('precipitation', 0)
    
    if row.get('is_snowing') == 1:
        m3.metric("Weather", "❄️ Snow", delta=f"{temp:.1f}°C", delta_color="inverse")
    elif row.get('is_raining') == 1:
        m3.metric("Weather", "🌧️ Rain", delta=f"{temp:.1f}°C", delta_color="normal")
    else:
        m3.metric("Weather", "☀️ Clear", delta=f"{temp:.1f}°C", delta_color="off")

    # 4. Glare Risk Metric
    if row.get('sun_glare') == 1:
        m4.metric("Visibility", "⚠️ High Glare", delta="Use Caution", delta_color="inverse")
    else:
        m4.metric("Visibility", "Good", delta="Normal")

# Best time to leave recommendation
recommendation = get_best_time_to_leave(future_data, wait_penalty_per_hour=patience)

if recommendation and recommendation['reduction'] > 50:
    from engine import calculate_commute_impact
    impact = calculate_commute_impact(recommendation['reduction'])

    # Determine the weather note
    is_hazard = recommendation.get('weather_hazard', False)
    weather_note = "⚠️ **Note:** AI suggests waiting for improved weather/visibility conditions." if is_hazard else ""

    st.success(f"""
        💡 **AI Commute Tip:** Leaving at **{recommendation['time']}** could save you from roughly **{recommendation['reduction']}** vehicles.
        
        {weather_note}
        
        ⏱️ **Estimated Savings:** ~{impact['mins']} minutes and ${impact['money']} in fuel costs.
    """)
elif recommendation:
    st.info(f"🕒 Traffic is currently at its lowest point for the next few hours. Now is the best time to leave!")

# --- 6. Visualization & AI Performance ---
col_left, col_right = st.columns([5, 3])

with col_left:
    st.subheader("🔮 24-Hour Traffic Forecast")
    fig = build_traffic_chart(history_data, future_data, selected_date)
    st.plotly_chart(fig, width="stretch")

with col_right:
    st.subheader("📊 AI Performance")
    improvement = ((baseline_mae - ai_mae) / baseline_mae) * 100 if baseline_mae != 0 else 0
    st.write(f"**Accuracy:** {ai_mae:.2f} MAE ({improvement:.1f}% boost)")
    
    # Feature Importance Chart (Great for hackathons!)
    fi_fig = build_feature_importance_chart(feature_importances) 
    st.plotly_chart(fi_fig, width="stretch")

# --- 7. Bottom Row: Live Spatial View (Map) ---
st.markdown("---")
st.subheader("🗺️ Live Camera Map")

camera_points = load_camera_points(allowed_locations=list_camera_names())
camera_names = [point["location"] for point in camera_points]
camera_name_set = set(camera_names)

if "selected_camera" not in st.session_state:
    st.session_state.selected_camera = None

if "map_dark_mode" not in st.session_state:
    st.session_state.map_dark_mode = False

if st.session_state.selected_camera not in camera_name_set:
    st.session_state.selected_camera = None


# --- Camera-to-Segment Sync Logic ---
import re
def _normalize_tokens(text):
    ROAD_STOP_WORDS = {
        "road", "rd", "street", "st", "avenue", "ave", "boulevard", "blvd",
        "highway", "hwy", "route", "rt", "interstate", "i", "us", "north",
        "south", "east", "west", "n", "s", "e", "w",
    }
    if not text:
        return set()
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    tokens = {token for token in raw_tokens if token and token not in ROAD_STOP_WORDS and len(token) > 1}
    return tokens


# Improved fuzzy matching for camera-to-segment
import difflib
def find_best_matching_segment(camera_name, segments):
    cam_tokens = _normalize_tokens(camera_name)
    cam_str = str(camera_name).lower()
    # Improved: extract main street name from segment and check as word in camera name
    for seg in segments:
        seg_tokens = _normalize_tokens(seg)
        if not seg_tokens:
            continue
        main_street = sorted(seg_tokens, key=len, reverse=True)[0]  # Longest token, usually the street name
        # Use regex to match as a word, ignore case
        if re.search(rf'\\b{re.escape(main_street)}\\b', cam_str, re.IGNORECASE):
            return seg
    # Fallback: fuzzy match on tokens
    best = None
    best_score = 0
    for seg in segments:
        seg_tokens = _normalize_tokens(seg)
        score = len(cam_tokens & seg_tokens)
        if score > best_score:
            best = seg
            best_score = score
    # Fallback: difflib fuzzy match if no token overlap
    if not best and segments:
        close = difflib.get_close_matches(camera_name, segments, n=1, cutoff=0.6)
        if close:
            return close[0]
    return best if best_score > 0 else None

camera_selector_options = ["No camera"] + camera_names
current_camera_option = st.session_state.selected_camera if st.session_state.selected_camera in camera_name_set else "No camera"

if st.session_state.get("camera_selector") != current_camera_option:
    st.session_state["camera_selector"] = current_camera_option

picked_camera_option = st.selectbox(
    "Active Camera",
    camera_selector_options,
    index=camera_selector_options.index(current_camera_option),
    key="camera_selector",
)



picked_camera = None if picked_camera_option == "No camera" else picked_camera_option
if picked_camera != st.session_state.selected_camera:
    st.session_state.selected_camera = picked_camera

# Always sync segment to selected camera if possible (even after rerun), but only rerun if segment changes
if st.session_state.selected_camera:
    match = find_best_matching_segment(st.session_state.selected_camera, all_segments)
    if match and match != st.session_state['selected_segment']:
        st.session_state['_pending_selected_segment'] = match
        st.rerun()

road_reference_row = segment_df[segment_df['DateTime'] == selected_date]
if road_reference_row.empty:
    road_reference_row = segment_df.head(1)

road_latitude = None
road_longitude = None
if not road_reference_row.empty:
    road_latitude = float(road_reference_row['gps_latitude'].iloc[0])
    road_longitude = float(road_reference_row['gps_longitude'].iloc[0])

st.markdown("### Cameras Along This Road")
along_road = get_cameras_along_road(camera_points, selected_segment)
if along_road:
    for idx, item in enumerate(along_road):
        camera_name = item["camera"].get("location", "Unknown camera")
        if st.button(camera_name, key=f"along_road_camera_{idx}_{camera_name}", width="stretch"):
            st.session_state.selected_camera = camera_name
            # Sync segment to camera
            match = find_best_matching_segment(camera_name, all_segments)
            if match and match != st.session_state['selected_segment']:
                st.session_state['_pending_selected_segment'] = match
            st.rerun()
else:
    st.caption("No named camera matches found for this road.")

st.markdown("### Cameras Near This Road")
near_road = get_cameras_near_road(camera_points, road_latitude, road_longitude)
if near_road:
    for idx, item in enumerate(near_road):
        camera_name = item["camera"]["location"]
        label = f"{camera_name} ({item['direction']}, {item['miles']:.1f} mi)"
        if st.button(label, key=f"near_road_camera_{idx}_{camera_name}", width="stretch"):
            st.session_state.selected_camera = camera_name
            # Sync segment to camera
            match = find_best_matching_segment(camera_name, all_segments)
            if match and match != st.session_state['selected_segment']:
                st.session_state['_pending_selected_segment'] = match
            st.rerun()
else:
    st.caption("No nearby camera points found.")

toggle_col_1, toggle_col_2 = st.columns([1, 1])
with toggle_col_2:
    live_update = st.checkbox("Live update selected camera", value=True, key="live_camera_updates")

prefer_native_html_stream = st.checkbox(
    "Use native HTML processed stream rendering (beta)",
    value=True,
    key="native_html_stream",
    help="Renders YOLO-processed frames via local MJPEG stream to reduce Streamlit flicker.",
)

selected_camera = st.session_state.selected_camera

LIVE_FRAME_REFRESH_INTERVAL = "80ms"
LIVE_STATS_REFRESH_INTERVAL = "450ms"


def render_embedded_camera_stream(stream_url, height=560, use_image_tag=False):
    safe_stream_url = escape(stream_url or "", quote=True)

    if use_image_tag:
        components.html(
            f"""
            <div style="width:100%;height:{height}px;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden;">
                <img src="{safe_stream_url}"
                     alt="Live camera stream"
                     style="width:100%;height:100%;object-fit:contain;background:#000;" />
            </div>
            """,
            height=height,
            scrolling=False,
        )
        return

    components.html(
        f"""
        <div style="width:100%;height:{height}px;background:#000;display:flex;align-items:center;justify-content:center;">
            <video id="camera-video" autoplay muted playsinline controls
                 style="width:100%;height:100%;object-fit:contain;background:#000;"></video>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        <script>
            (function() {{
                const url = "{safe_stream_url}";
                const video = document.getElementById("camera-video");
                if (!url || !video) return;

                const lower = url.toLowerCase();
                const isLikelyHls = lower.includes('.m3u8');

                if (isLikelyHls && window.Hls && window.Hls.isSupported()) {{
                    const hls = new Hls({{ enableWorker: true, lowLatencyMode: true, backBufferLength: 5 }});
                    hls.loadSource(url);
                    hls.attachMedia(video);
                    hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                        video.play().catch(function() {{}});
                    }});
                }} else if (video.canPlayType('application/vnd.apple.mpegurl') || !isLikelyHls) {{
                    video.src = url;
                    video.play().catch(function() {{}});
                }}
            }})();
        </script>
        """,
        height=height,
        scrolling=False,
    )


@st.fragment(run_every=LIVE_FRAME_REFRESH_INTERVAL)
def render_live_camera_frame():
    active_camera = st.session_state.selected_camera
    if not active_camera:
        st.info("Click a camera on the map to start a worker and view its live feed.")
        return

    frame_bytes, _ = get_worker_snapshot(active_camera)

    last_frame_key = f"camera_last_frame::{active_camera}"
    if frame_bytes:
        st.session_state[last_frame_key] = frame_bytes

    display_frame = frame_bytes or st.session_state.get(last_frame_key)
    if display_frame:
        st.image(display_frame, channels="BGR", width=900)
    else:
        st.info("Worker started. Waiting for camera frames...")


@st.fragment(run_every=LIVE_STATS_REFRESH_INTERVAL)
def render_live_camera_stats():
    active_camera = st.session_state.selected_camera
    if not active_camera:
        return

    _, camera_stats = get_worker_snapshot(active_camera)

    selected_from_stats_nearby = render_camera_stats(camera_stats, camera_points, key_prefix="camera_stats")
    if selected_from_stats_nearby and selected_from_stats_nearby != st.session_state.selected_camera:
        st.session_state.selected_camera = selected_from_stats_nearby
        st.rerun()


if live_update:
    active_camera = st.session_state.selected_camera
    if active_camera:
        processed_stream_url = get_processed_stream_url(active_camera)

        st.markdown(f"### 📷 Camera Feed: {active_camera}")
        if prefer_native_html_stream and isinstance(processed_stream_url, str) and processed_stream_url.startswith(("http://", "https://")):
            render_embedded_camera_stream(processed_stream_url, use_image_tag=True)
        else:
            with st.container(key="live_camera_frame"):
                st.markdown(
                    """
                    <style>
                    .st-key-live_camera_frame [data-testid="stImage"],
                    .st-key-live_camera_frame [data-testid="stImage"] img {
                        animation: none !important;
                        transition: none !important;
                        opacity: 1 !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                render_live_camera_frame()
        render_live_camera_stats()
    else:
        st.info("Click a camera on the map to start a worker and view its live feed.")
else:
    active_camera = st.session_state.selected_camera
    if active_camera:
        frame_bytes, camera_stats = get_worker_snapshot(active_camera)
        st.markdown(f"### 📷 Camera Feed: {active_camera}")
        if frame_bytes:
            st.image(frame_bytes, channels="BGR", width=900)
        else:
            st.info("Worker started. Waiting for camera frames...")
        selected_from_stats_nearby = render_camera_stats(camera_stats, camera_points, key_prefix="camera_stats_static")
        if selected_from_stats_nearby and selected_from_stats_nearby != st.session_state.selected_camera:
            st.session_state.selected_camera = selected_from_stats_nearby
            st.rerun()
    else:
        st.info("Click a camera on the map to start a worker and view its live feed.")

map_header_left, map_header_right = st.columns([8, 1])
with map_header_left:
    st.markdown("### Pick From Map")
with map_header_right:
    st.button(
        "☾" if not st.session_state.map_dark_mode else "☼",
        key="map_dark_mode_button",
        help="Toggle map light/dark mode",
        width="stretch",
        on_click=lambda: setattr(st.session_state, "map_dark_mode", not st.session_state.map_dark_mode),
    )

if camera_points:
    map_figure = build_camera_map_figure(
        camera_points,
        selected_camera=st.session_state.selected_camera,
    )
    with st.container(key="camera_map_shell"):
        map_filter_css = "invert(1) hue-rotate(180deg) saturate(0.85) brightness(0.9)" if st.session_state.map_dark_mode else "none"
        st.markdown(
            f"""
            <style>
            .st-key-camera_map_shell [data-testid="stPlotlyChart"] {{
                filter: {map_filter_css};
                transition: filter 120ms ease-in-out;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        map_selection = st.plotly_chart(
            map_figure,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key="camera_map",
        )
    selected_from_map = extract_selected_camera_from_map_event(map_selection)
    if selected_from_map and selected_from_map != st.session_state.selected_camera:
        st.session_state.selected_camera = selected_from_map
        st.rerun()
else:
    st.warning("No camera map points are currently available.")