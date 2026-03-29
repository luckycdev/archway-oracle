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

# Time Logic & Slider (Fixed Syntax)
now = datetime.now()
start_of_today = datetime(now.year, now.month, now.day, 0, 0)
end_of_today = datetime(now.year, now.month, now.day, 23, 0)
current_hour_val = datetime(now.year, now.month, now.day, now.hour, 0)

st.sidebar.markdown(f"**📅 Date:** {now.strftime('%B %d, %Y')}")
selected_date = st.sidebar.slider(
    "2. Set Prediction Time",
    min_value=start_of_today,
    max_value=end_of_today,
    value=current_hour_val,
    step=timedelta(hours=1),
    format="h:mm A"
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
    st.success(f"💡 **AI Commute Tip:** Leaving at **{recommendation['time']}** could save you from roughly **{recommendation['reduction']}** vehicles.")
elif recommendation:
    st.info(f"✨ **Note:** Traffic is currently stable. No significant drops expected in the next 3 hours.")

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
    all_segments_at_time = test_results[test_results['DateTime'].dt.hour == selected_date.hour]
    build_google_map(all_segments_at_time, selected_segment, api_key, secondary_segment) 
else:
    st.info("💡 Pro Tip: Add a Google Maps API key to secrets.toml.")