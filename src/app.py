import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# Import from our local src modules
from data_processing import load_and_prep_data, classify_traffic
from model import train_and_evaluate
from visualizations import build_traffic_chart, build_feature_importance_chart
from maps import build_google_map, show_map_legend

# --- 1. Page Configuration ---
st.set_page_config(page_title="St. Louis Traffic AI", layout="wide", page_icon="🚦")
st.title("🚦 St. Louis Smart City Traffic Dashboard")

# --- 2. Load Data & Model ---
@st.cache_resource
def get_data_and_model():
    # Load the processed St. Louis data
    # Note: Ensure this path matches your folder structure
    df = load_and_prep_data("C:/Users/flami/Repositories/Traffic Predictor/traffic-predictor/TrafficCounts2024_-8820105916022295803.csv")
    
    # Train the model on the synthesized hourly patterns
    test_data, ai_mae, baseline_mae, feature_importances, winning_params = train_and_evaluate(df)
    
    return df, test_data, ai_mae, baseline_mae, feature_importances, winning_params

with st.spinner("Analyzing St. Louis Road Network..."):
    full_data, test_results, ai_mae, baseline_mae, feature_importances, winning_params = get_data_and_model()

top_metrics_container = st.sidebar.container()

st.sidebar.markdown("---")
st.sidebar.header("🕹️ Control Panel")

# Select Road Segment
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

# Time Logic & Slider
now = datetime.now()
start_of_today = datetime(now.year, now.month, now.day, 0, 0)
end_of_today = datetime(now.year, now.month, now.day, 23, 0)
current_hour = datetime(now.year, now.month, now.day, now.hour, 0)

st.sidebar.markdown(f"**📅 Date:** {now.strftime('%B %d, %Y')}")
selected_date = st.sidebar.slider(
    "2. Set Prediction Time",
    min_value=start_of_today,
    max_value=end_of_today,
    value=current_hour,
    step=timedelta(hours=1),
    format="h:mm A"
)

# --- 4. Logic: Data Filtering ---
segment_df = test_results[test_results['road_segment_id'] == selected_segment].sort_values('DateTime')
history_data = segment_df[segment_df['DateTime'] <= selected_date]
future_data = segment_df[segment_df['DateTime'] > selected_date].copy()

# --- 5. Populate the Top Sidebar Metrics ---
current_row = segment_df[segment_df['DateTime'] == selected_date]

if not current_row.empty:
    row = current_row.iloc[0]
    
    # 2. Teleport the data back up into the empty container!
    with top_metrics_container:
        st.markdown(f"### 📍 {selected_segment}")
        
        # Traffic Stats (Using columns to make a neat 2x2 grid in the sidebar)
        st.markdown("**🚗 Traffic Pulse**")
        t1, t2 = st.columns(2)
        t1.metric("Actual Vol", f"{int(row['vehicle_count'])}/hr")
        t2.metric("AI Forecast", f"{int(row['Predicted_Vehicles'])}/hr")
        st.metric("Status", row['Traffic_Level'])
        
        st.markdown("---")
        
        # Environment Stats
        st.markdown("**🌍 Environment**")
        e1, e2 = st.columns(2)
        e1.metric("Temp", f"{row['temperature']:.1f} °C")
        e2.metric("Rain", f"{row['precipitation']:.1f} mm")
        
        e3, e4 = st.columns(2)
        e3.metric("Clouds", f"{row['cloud_cover']:.0f}%")
        e4.metric("Glare", "Yes ⚠️" if row['sun_glare'] == 1 else "No")
col_left, col_right = st.columns([5, 3]) # Gave slightly more room to the right column

with col_left:
    st.subheader("🔮 24-Hour Traffic Forecast")
    fig = build_traffic_chart(history_data, future_data, selected_date)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📊 AI Performance")
    
    # 1. High-level metrics
    improvement = ((baseline_mae - ai_mae) / baseline_mae) * 100 if baseline_mae != 0 else 0
    st.write(f"**Model Accuracy:** {ai_mae:.2f} MAE")
    st.write(f"**Improvement:** {improvement:.1f}% over baseline")
    
    # 2. Feature Importance Chart
    st.markdown("**What drives the AI's predictions?**")
    fi_fig = build_feature_importance_chart(feature_importances) 
    st.plotly_chart(fi_fig, use_container_width=True)
    
    # 3. Keep the JSON hidden cleanly in an expander
    with st.expander("Technical Parameters"):
        st.json(winning_params)

# --- 7. Bottom Row: Live Spatial View (Map) ---
st.markdown("---")
st.subheader("🗺️ Live Spatial View")
show_map_legend()

api_key = st.secrets.get("GOOGLE_MAPS_API_KEY")
if api_key:
    # We now filter the FULL results for the specific hour selected
    all_segments_at_time = test_results[test_results['DateTime'].dt.hour == selected_date.hour]
    
    # The map will now take up the full width of the screen at the bottom
    build_google_map(all_segments_at_time, selected_segment, api_key)
else:
    st.info("💡 Pro Tip: Add a Google Maps API key to secrets.toml.")
