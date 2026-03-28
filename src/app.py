import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Import from our src modules
from src.data_processing import load_and_prep_data, classify_traffic
from src.model import train_and_evaluate
from src.visualizations import build_traffic_chart
from src.maps import build_google_map, show_map_legend

# --- 1. Page Configuration
st.set_page_config(page_title="Traffic Predictor AI", layout="wide", page_icon="🚦")
st.title("🚦 Smart City Traffic Prediction Dashboard")
st.markdown("Predicting future traffic jams by analyzing historical trends, GPS location, and real-time weather conditions.")

# --- 2. Load Data & Model ---
@st.cache_resource
def get_data():
    df = load_and_prep_data()
    # Downcast floats to 32-bit to save 50% RAM on those specific columns
    for col in ['gps_latitude', 'gps_longitude', 'hour_sin', 'hour_cos']:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df

@st.cache_resource
def get_model(df):
    # Only train on the last 100,000 rows to save memory
    train_subset = df.tail(300000).copy()
    return train_and_evaluate(train_subset)

with st.spinner("Processing 2024 TrafficTab Data..."):
    data = get_data()

with st.spinner("Training AI Model (HistGradientBoosting)..."):
    test_data, ai_mae, baseline_mae, cv_scores, winning_params = get_model(data)

# --- 3. Sidebar ---
st.sidebar.header("🕹️ Controls")
segments = sorted(test_data['road_segment_id'].unique())
selected_segment = st.sidebar.selectbox("1. Select a Road Segment", segments)

# Filter data for the specific segment
segment_data = test_data[test_data['road_segment_id'] == selected_segment].sort_values('DateTime')

if not segment_data.empty:
    min_available = segment_data['DateTime'].min() + pd.Timedelta(hours=12)
    max_available = segment_data['DateTime'].max() - pd.Timedelta(hours=1)

    if min_available < max_available:
        selected_date = st.sidebar.slider(
            "2. Pick the 'Present' Moment",
            min_value=min_available.to_pydatetime(),
            max_value=max_available.to_pydatetime(),
            value=min_available.to_pydatetime(),
            format="MM-DD HH:mm"
        )
    else:
        selected_date = segment_data['DateTime'].max() - pd.Timedelta(minutes=30)
        st.sidebar.warning("Limited time window for this segment.")

    # --- 4. Filter Data for Charting ---
    history_data = segment_data[
        (segment_data['DateTime'] <= selected_date) & 
        (segment_data['DateTime'] >= selected_date - pd.Timedelta(hours=24))
    ]
    future_data = segment_data[
        (segment_data['DateTime'] > selected_date) & 
        (segment_data['DateTime'] <= selected_date + pd.Timedelta(hours=12))
    ].copy()

    # Fill any stray NaNs with 0 or a neutral value
    future_data['Predicted_Vehicles'] = future_data['Predicted_Vehicles'].fillna(0)

    # --- 5. Context Panel ---
    st.markdown(f"### 🧠 AI Context for Segment {selected_segment}")
    current_context = segment_data[segment_data['DateTime'] == selected_date]
    
    if not current_context.empty:
        ctx = current_context.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        # Using new columns from TrafficTab23
        c1.metric("☁️ Weather", str(ctx.get('weather_condition', 'N/A')))
        c2.metric("🛣️ Road Status", str(ctx.get('road_surface_status', 'Normal')))
        c3.metric("🎉 Holiday?", "Yes" if ctx['is_holiday'] == 1 else "No")
        c4.metric("📍 Coordinates", f"{ctx['gps_latitude']:.3f}, {ctx['gps_longitude']:.3f}")

    # --- 6. Chart & Map ---
    col_chart, col_map = st.columns([2, 1])

    with col_chart:
        st.markdown("#### 🔮 12-Hour Forecast")
        fig = build_traffic_chart(history_data, future_data, selected_date)
        st.plotly_chart(fig, use_container_width=True)

    with col_map:
        st.markdown("#### 🗺️ Live Spatial View")
        show_map_legend()
        api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", None)
        if api_key:
            build_google_map(future_data, selected_segment, api_key)
        else:
            st.warning("Google Maps API Key missing in secrets.toml.")
        
    # --- 7. Travel Recommendations ---
    st.markdown("---")
    res1, res2 = st.columns(2)
    
    with res1:
        if not future_data.empty:
            next_hour = future_data.iloc[0]
            status = next_hour['Traffic_Level']
            st.subheader("🚨 Next Hour Status")
            if "Red" in status: st.error(status)
            elif "Yellow" in status: st.warning(status)
            else: st.success(status)
            
    with res2:
        if not future_data.empty:
            best_time_row = future_data.loc[future_data['Predicted_Vehicles'].idxmin()]
            st.subheader("🚗 Best Time to Leave")
            st.info(f"Recommended: **{best_time_row['DateTime'].strftime('%I:%M %p')}**")

    # --- 8. Performance Metrics ---
    st.markdown("---")
    st.subheader("🏆 Model Reliability")
    
    # Calculate improvement vs baseline (lag1)
    improvement = ((baseline_mae - ai_mae) / baseline_mae) * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Naive Baseline Error", f"{baseline_mae:.1f} cars")
    m2.metric("AI Model Error (MAE)", f"{ai_mae:.1f} cars", delta=f"{improvement:.1f}% Better")
    
    # Local segment accuracy
    seg_mae = mean_absolute_error(segment_data['vehicle_count'], segment_data['Predicted_Vehicles'])
    m3.metric("This Segment Accuracy", f"{seg_mae:.1f} cars")

    # --- 9. Tech Details ---
    with st.expander("🛠️ View Model Parameters"):
        st.write("Winning parameters from Hyperparameter Tuning:")
        st.json(winning_params)

else:
    st.error("No data found for this road segment. Check your preprocessing steps.")