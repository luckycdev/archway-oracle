import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import time 

# Import from our local src modules
from src.data_processing import load_and_prep_data, classify_traffic
from src.model import train_and_evaluate
from src.visualizations import build_traffic_chart, build_feature_importance_chart # Added this
from src.maps import build_google_map, show_map_legend
from src.engine import get_best_time_to_leave

# --- 1. Page Configuration ---
st.set_page_config(page_title="St. Louis Traffic AI", layout="wide", page_icon="🚦")
st.title("🚦 St. Louis Smart City Traffic Dashboard")

# --- 2. Load Data & Model ---
@st.cache_resource
def get_data_and_model():
    df = load_and_prep_data("data/stl_traffic_counts.csv")
    # Added feature_importances to the return
    test_data, ai_mae, baseline_mae, cv_scores, winning_params, feature_importances = train_and_evaluate(df)
    return df, test_data, ai_mae, baseline_mae, winning_params, feature_importances

with st.spinner("Analyzing St. Louis Road Network..."):
    full_data, test_results, ai_mae, baseline_mae, winning_params, feature_importances = get_data_and_model()

# --- 3. Sidebar Controls ---
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
recommendation = get_best_time_to_leave(future_data)
if recommendation and recommendation['reduction'] > 50:
    from src.engine import calculate_commute_impact
    impact = calculate_commute_impact(recommendation['reduction'])

    # Special weather warning if the AI is dodging a storm
    if recommendation['weather_hazard']:
        weather_note = "⚠️ **Note:** AI suggests waiting for improved weather/visiblity conditions."

    st.success(f"""
        💡 **AI Commute Tip:** Leaving at **{recommendation['time']}** could save you from roughly **{recommendation['reduction']}** vehicles.
        {weather_note if recommendation.get('weather_hazard') else ""}
        
        ⏱️ **Estimated Savings:** ~{impact['mins']} minutes and ${impact['money']} in fuel costs.
    """)
elif recommendation:
    st.info(f"🕒 Traffic is currently at its lowest point for the next few hours. Now is the best time to leave!")

# --- 6. Visualization & AI Performance ---
col_left, col_right = st.columns([5, 3])

with col_left:
    st.subheader("🔮 24-Hour Traffic Forecast")
    fig = build_traffic_chart(history_data, future_data, selected_date)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📊 AI Performance")
    improvement = ((baseline_mae - ai_mae) / baseline_mae) * 100 if baseline_mae != 0 else 0
    st.write(f"**Accuracy:** {ai_mae:.2f} MAE ({improvement:.1f}% boost)")
    
    # Feature Importance Chart (Great for hackathons!)
    fi_fig = build_feature_importance_chart(feature_importances) 
    st.plotly_chart(fi_fig, use_container_width=True)

# --- 7. Bottom Row: Live Spatial View (Map) ---
st.markdown("---")
st.subheader("🗺️ Live Spatial View")
show_map_legend()

api_key = st.secrets.get("GOOGLE_MAPS_API_KEY")
if api_key:
    all_segments_at_time = test_results[test_results['DateTime'] == selected_date]
    build_google_map(all_segments_at_time, selected_segment, api_key, secondary_segment) 
else:
    st.info("💡 Pro Tip: Add a Google Maps API key to secrets.toml.")