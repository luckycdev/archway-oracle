import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Import from our local src modules
from src.data_processing import load_and_prep_data, classify_traffic
from src.model import train_and_evaluate
from src.visualizations import build_traffic_chart
from src.maps import build_google_map, show_map_legend

# --- 1. Page Configuration ---
st.set_page_config(page_title="St. Louis Traffic AI", layout="wide", page_icon="🚦")
st.title("🚦 St. Louis Smart City Traffic Dashboard")
st.markdown("""
    **Location:** St. Louis County, MO  
    Predicting traffic flow using MoDOT-aligned AWT data and Gradient Boosting AI.
""")

# --- 2. Load Data & Model ---
@st.cache_resource
def get_data_and_model():
    # Load the processed St. Louis data
    df = load_and_prep_data("/workspaces/traffic-predictor/data/stl_traffic_counts.csv")
    
    # Train the model on the synthesized hourly patterns
    test_data, ai_mae, baseline_mae, cv_scores, winning_params = train_and_evaluate(df)
    
    return df, test_data, ai_mae, baseline_mae, winning_params

with st.spinner("Analyzing St. Louis Road Network..."):
    full_data, test_results, ai_mae, baseline_mae, winning_params = get_data_and_model()

# --- 3. Sidebar Controls ---
st.sidebar.header("🕹️ Control Panel")

# Select Road Segment (using the 'Onstreet' names from your CSV)
all_segments = sorted(test_results['road_segment_id'].unique())
selected_segment = st.sidebar.selectbox("1. Select a Roadway", all_segments)

# Filter results for the specific road
segment_df = test_results[test_results['road_segment_id'] == selected_segment].sort_values('DateTime')

# Time Slider (March 28, 2026)
# We set the default to 4:00 PM (16:00) to show the evening rush hour
default_time = datetime(2026, 3, 28, 16, 0)
selected_date = st.sidebar.slider(
    "2. Set Prediction Time",
    min_value=datetime(2026, 3, 28, 0, 0),
    max_value=datetime(2026, 3, 28, 23, 0),
    value=default_time,
    format="HH:mm"
)

# --- 4. Logic: History vs Future ---
history_data = segment_df[segment_df['DateTime'] <= selected_date]
future_data = segment_df[segment_df['DateTime'] > selected_date].copy()

# --- 5. Key Metrics Display ---
st.markdown(f"### 📍 Real-Time Analysis: {selected_segment}")
current_row = segment_df[segment_df['DateTime'] == selected_date]

if not current_row.empty:
    row = current_row.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Volume", f"{int(row['vehicle_count'])} cars/hr")
    m2.metric("AI Prediction", f"{int(row['Predicted_Vehicles'])} cars/hr")
    m3.metric("Status", row['Traffic_Level'])
    m4.metric("Coordinates", f"{row['gps_latitude']:.3f}, {row['gps_longitude']:.3f}")

# --- 6. Visualization & Map ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🔮 24-Hour Traffic Forecast")
    fig = build_traffic_chart(history_data, future_data, selected_date)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🗺️ Geographic View")
    show_map_legend()
    # Pull Google Maps API Key from secrets
    api_key = st.secrets.get("GOOGLE_MAPS_API_KEY")
    if api_key:
        # Pass future data to see upcoming traffic on the map
        build_google_map(future_data, selected_segment, api_key)
    else:
        st.info("💡 Pro Tip: Add a Google Maps API key to secrets.toml to enable the interactive map.")

# --- 7. Model Insights ---
st.markdown("---")
with st.expander("📊 AI Performance & Technical Details"):
    c1, c2 = st.columns(2)
    improvement = ((baseline_mae - ai_mae) / baseline_mae) * 100 if baseline_mae != 0 else 0
    c1.write(f"**Model Accuracy (MAE):** {ai_mae:.2f} vehicles")
    c1.write(f"**Improvement over Baseline:** {improvement:.1f}%")
    c2.write("**Winning AI Parameters:**")
    c2.json(winning_params)