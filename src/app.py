import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from html import escape
from urllib.parse import urlsplit, urlunsplit

# Import from our local src modules
from data_processing import (
    MONTHLY_FACTORS,
    calculate_aadt_estimate,
    calculate_daily_traffic,
    load_and_prep_data,
    classify_traffic,
)
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
from camera_workers import (
    ensure_background_camera_sampler,
    get_camera_background_status,
    get_camera_display_name,
    get_processed_stream_url,
    get_worker_snapshot,
    list_camera_names,
    reload_camera_sources,
    resolve_camera_name,
    set_prediction_context,
    get_yolo_model,
)
from config import (
    APP_PAGE_ICON,
    APP_PAGE_LAYOUT,
    APP_PAGE_TITLE,
    CAMERA_BACKGROUND_DWELL_SECONDS,
    CAMERA_BACKGROUND_SAMPLE_SECONDS,
    CAMERA_BACKGROUND_SCAN_ENABLED,
    CAMERA_BACKGROUND_WORKERS,
    LIVE_STATS_REFRESH_INTERVAL,
    PATIENCE_DEFAULT,
    PATIENCE_MAX,
    PATIENCE_MIN,
    PATIENCE_STEP,
    PROCESSED_STREAM_PORT,
)

# --- 1. Page Configuration ---
st.set_page_config(page_title=APP_PAGE_TITLE, layout=APP_PAGE_LAYOUT, page_icon=APP_PAGE_ICON)
st.title(f"{APP_PAGE_ICON} {APP_PAGE_TITLE}")

# --- 2. Load Data & Model ---
@st.cache_resource
def get_data_and_model():
    df = load_and_prep_data()
    # Added feature_importances to the return
    test_data, ai_mae, baseline_mae, cv_scores, winning_params, feature_importances = train_and_evaluate(df)
    return df, test_data, ai_mae, baseline_mae, winning_params, feature_importances

with st.spinner("Analyzing St. Louis Road Network..."):
    full_data, test_results, ai_mae, baseline_mae, winning_params, feature_importances = get_data_and_model()

# --- 3. Initialize YOLO Model with Device Selection ---
@st.cache_resource
def init_yolo_model():
    print("\n" + "="*70, flush=True)
    print("INITIALIZING YOLO MODEL ON APP STARTUP", flush=True)
    print("="*70, flush=True)
    model = get_yolo_model()
    print("="*70, flush=True)
    return model

with st.spinner("Initializing Computer Vision Model..."):
    init_yolo_model()

ensure_background_camera_sampler()

# --- 4. Sidebar Controls ---
st.sidebar.header("🕹️ Control Panel")

# Select Primary Roadway
all_segments = sorted(test_results['road_segment_id'].unique())
selected_segment = st.sidebar.selectbox("1. Select Primary Roadway", all_segments)

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
    min_value=PATIENCE_MIN,
    max_value=PATIENCE_MAX,
    value=PATIENCE_DEFAULT,
    step=PATIENCE_STEP,
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

    st.markdown("---")
    st.subheader("📊 Annual Average Daily Traffic")

    daily_vol = calculate_daily_traffic(full_data, selected_segment, selected_day)
    aadt_val = calculate_aadt_estimate(daily_vol, selected_day)
    current_factor = MONTHLY_FACTORS.get(selected_day.month, 1.0)

    col_stats1, col_stats2, col_stats3 = st.columns(3)

    with col_stats1:
        st.metric("Daily Volume", f"{daily_vol:,} cars")
        st.caption(f"Total for {selected_day.strftime('%b %d')}")

    with col_stats2:
        st.metric("Est. AADT", f"{aadt_val:,} cars")
        st.caption("Normalized Annual Average")

    with col_stats3:
        if aadt_val > 0:
            index = daily_vol / aadt_val
            status = "Above Avg" if index > 1.1 else "Below Avg" if index < 0.9 else "Normal"
            st.metric(
                label="Seasonal Load",
                value=f"{current_factor}x",
                delta=f"{status} ({index:.2f}x)",
                delta_color="inverse",
            )
        else:
            st.metric("Seasonal Load", "N/A")

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
def render_camera_background_ui():
    st.subheader("🗺️ Live Camera Map")

    _, refresh_col_2 = st.columns([6, 2])
    with refresh_col_2:
        if st.button("Refresh camera list", key="refresh_camera_list_button", width="stretch"):
            reload_camera_sources()
            st.rerun()

    camera_names = list_camera_names()
    camera_points = load_camera_points(allowed_locations=camera_names)
    camera_display_map = {camera_name: get_camera_display_name(camera_name) for camera_name in camera_names}
    display_to_camera = {display_name: camera_name for camera_name, display_name in camera_display_map.items()}
    camera_name_set = set(camera_names)

    if "selected_camera" not in st.session_state:
        st.session_state.selected_camera = None

    if "selected_camera_source" not in st.session_state:
        st.session_state.selected_camera_source = None

    if "camera_map_key_version" not in st.session_state:
        st.session_state.camera_map_key_version = 0

    if "map_dark_mode" not in st.session_state:
        st.session_state.map_dark_mode = False

    if st.session_state.selected_camera not in camera_name_set:
        st.session_state.selected_camera = None

    camera_selector_options = ["No camera"] + [camera_display_map[camera_name] for camera_name in camera_names]
    selector_value = camera_display_map.get(st.session_state.selected_camera, "No camera")
    if st.session_state.get("camera_selector") != selector_value:
        st.session_state.camera_selector = selector_value

    def _on_camera_selector_change():
        selected_option = st.session_state.get("camera_selector", "No camera")
        selected_name = None
        if selected_option != "No camera":
            selected_name = display_to_camera.get(selected_option, resolve_camera_name(selected_option))
        if selected_name != st.session_state.selected_camera:
            st.session_state.selected_camera = selected_name
            st.session_state.selected_camera_source = "dropdown"
            # Force a fresh map widget so persisted point selection cannot override dropdown choice.
            st.session_state.camera_map_key_version += 1

    st.selectbox(
        "Active Camera",
        camera_selector_options,
        key="camera_selector",
        on_change=_on_camera_selector_change,
    )

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
            display_name = get_camera_display_name(camera_name)
            if st.button(display_name, key=f"along_road_camera_{idx}_{camera_name}", width="stretch"):
                st.session_state.selected_camera = camera_name
                st.session_state.selected_camera_source = "road_list"
                st.rerun()
    else:
        st.caption("No named camera matches found for this road.")

    st.markdown("### Cameras Near This Road")
    near_road = get_cameras_near_road(camera_points, road_latitude, road_longitude, limit=5)
    if near_road:
        for idx, item in enumerate(near_road):
            camera_name = item["camera"]["location"]
            label = f"{get_camera_display_name(camera_name)} ({item['direction']}, {item['miles']:.1f} mi)"
            if st.button(label, key=f"near_road_camera_{idx}_{camera_name}", width="stretch"):
                st.session_state.selected_camera = camera_name
                st.session_state.selected_camera_source = "nearby_list"
                st.rerun()
    else:
        st.caption("No nearby camera points found.")

    def _get_request_hostname():
        try:
            headers = getattr(st.context, "headers", None)
            if headers:
                host = headers.get("host") or headers.get("Host")
                if host:
                    return host.split(":", 1)[0].strip("[]")
        except Exception:
            return None
        return None

    def _resolve_stream_url_for_client(stream_url):
        if not isinstance(stream_url, str) or not stream_url:
            return ""

        request_host = _get_request_hostname()

        if stream_url.startswith("/"):
            if request_host in {"localhost", "127.0.0.1", "0.0.0.0"}:
                return f"http://{request_host}:{PROCESSED_STREAM_PORT}{stream_url}"
            return stream_url

        try:
            parts = urlsplit(stream_url)
        except Exception:
            return stream_url

        if parts.hostname not in {"127.0.0.1", "localhost", "0.0.0.0"}:
            return stream_url

        if not request_host:
            return stream_url

        user_info = ""
        if parts.username:
            user_info = parts.username
            if parts.password:
                user_info = f"{user_info}:{parts.password}"
            user_info = f"{user_info}@"

        port_suffix = f":{parts.port}" if parts.port else ""
        rebuilt_netloc = f"{user_info}{request_host}{port_suffix}"
        return urlunsplit((parts.scheme, rebuilt_netloc, parts.path, parts.query, parts.fragment))

    def render_processed_stream_html(stream_url, height=560):
        safe_stream_url = escape(_resolve_stream_url_for_client(stream_url), quote=True)
        st.markdown(
            f"""
            <div style="width:100%;max-width:900px;height:{height}px;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden;">
                <img
                    id="processed-feed"
                    src="{safe_stream_url}"
                    alt="Processed camera feed"
                    style="width:100%;height:100%;object-fit:contain;background:#000;"
                />
            </div>
            """,
            unsafe_allow_html=True,
        )

    @st.fragment(run_every=LIVE_STATS_REFRESH_INTERVAL)
    def render_live_camera_stats():
        active_camera = st.session_state.selected_camera
        if not active_camera:
            return

        _, camera_stats = get_worker_snapshot(active_camera)

        selected_from_stats_nearby = render_camera_stats(camera_stats, camera_points, key_prefix="camera_stats")
        if selected_from_stats_nearby and selected_from_stats_nearby != st.session_state.selected_camera:
            st.session_state.selected_camera = selected_from_stats_nearby
            st.session_state.selected_camera_source = "stats_nearby"
            st.rerun()

    active_camera = st.session_state.selected_camera
    if active_camera:
        processed_stream_url = get_processed_stream_url(active_camera)
        st.markdown(f"### 📷 Camera Feed: {get_camera_display_name(active_camera)}")
        if isinstance(processed_stream_url, str) and processed_stream_url.startswith(("http://", "https://", "/")):
            render_processed_stream_html(processed_stream_url)
        else:
            st.info("Worker started. Waiting for processed stream URL...")
        render_live_camera_stats()
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
        camera_status_lookup = {point["location"]: get_camera_background_status(point["location"]) for point in camera_points}
        map_figure = build_camera_map_figure(
            camera_points,
            selected_camera=st.session_state.selected_camera,
            camera_statuses=camera_status_lookup,
            background_scan_enabled=CAMERA_BACKGROUND_SCAN_ENABLED,
        )
        map_key = f"camera_map_{st.session_state.camera_map_key_version}"

        def _on_camera_map_select():
            selection_state = st.session_state.get(map_key)
            selected_from_map = resolve_camera_name(extract_selected_camera_from_map_event(selection_state, camera_points))
            if selected_from_map and selected_from_map != st.session_state.selected_camera:
                st.session_state.selected_camera = selected_from_map
                st.session_state.selected_camera_source = "map"

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
                on_select=_on_camera_map_select,
                selection_mode="points",
                config={"scrollZoom": True, "doubleClick": False, "doubleClickDelay": 1000},
                key=map_key,
            )
    else:
        st.warning("No camera map points are currently available.")

    if CAMERA_BACKGROUND_SCAN_ENABLED:
        with st.expander("Debug: Background Camera Sampler", expanded=False):
            st.caption("Development-only visibility into background camera sampling state.")
            st.write(f"Enabled: {CAMERA_BACKGROUND_SCAN_ENABLED}")
            st.write(f"Sample seconds per camera: {CAMERA_BACKGROUND_SAMPLE_SECONDS}")
            st.write(f"Queue dwell seconds: {CAMERA_BACKGROUND_DWELL_SECONDS}")
            st.write(f"Configured workers: {CAMERA_BACKGROUND_WORKERS}")

            debug_rows = []
            for camera_name in camera_names:
                status = get_camera_background_status(camera_name)
                movement_counts = status.get("movement_counts", {}) or {}
                debug_rows.append(
                    {
                        "Camera": camera_name,
                        "Display": get_camera_display_name(camera_name),
                        "Badge": status.get("badge", ""),
                        "Traffic": status.get("traffic_label", "No Feed"),
                        "Ended Score": status.get("traffic_score"),
                        "Stopped": int(movement_counts.get("stopped", 0) or 0),
                        "Slow": int(movement_counts.get("slow", 0) or 0),
                        "Fast": int(movement_counts.get("fast", 0) or 0),
                        "Vehicles": int(status.get("vehicle_count", 0) or 0),
                        "Last Sampled": status.get("last_sampled", ""),
                    }
                )

            if debug_rows:
                st.dataframe(pd.DataFrame(debug_rows), width="stretch", hide_index=True)
            else:
                st.caption("No camera status data yet.")


render_camera_background_ui()