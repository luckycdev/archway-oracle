import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Import from our src modules
from src.data_processing import load_and_prep_data, classify_traffic
from src.model import train_and_evaluate
from src.visualizations import build_traffic_chart
from src.maps import build_google_map, show_map_legend

# --- 1. Page Configuration
st.set_page_config(page_title="Traffic Predictor AI", layout="wide")
st.title("🚦 Smart City Traffic Prediction Dashboard")
st.markdown("Predicting future traffic jams by analyzing historical trends, weather, and holidays.")

# --- 2. Load Data ---
@st.cache_resource
def get_data():
    return load_and_prep_data()

@st.cache_resource
def get_model(df):
    return train_and_evaluate(df)

with st.spinner("Preprocessing data & pulling St. Louis weather..."):
    data = get_data()

with st.spinner("Tuning Hyperparameters & Running Cross-Validation (This may take a minute)..."):
    test_data, ai_mae, baseline_mae, cv_scores, winning_params = get_model(data)

# --- 3. Sidebar ---
st.sidebar.header("Time Machine Controls")
st.sidebar.info("Use these controls to step into the past and see what the AI would have predicted.")

junctions = sorted(data['Junction'].unique())
selected_junction = st.sidebar.selectbox("1. Select a Traffic Junction", junctions)

junction_data = test_data[test_data['Junction'] == selected_junction].sort_values('DateTime')

if not junction_data.empty:
    min_date = junction_data['DateTime'].min() + pd.Timedelta(days=3)
    max_date = junction_data['DateTime'].max() - pd.Timedelta(hours=24)
    
    selected_date = st.sidebar.slider(
        "2. Pick the 'Present' Moment", 
        min_value=min_date.to_pydatetime(), max_value=max_date.to_pydatetime(), 
        value=min_date.to_pydatetime(), format="YYYY-MM-DD HH:mm"
    )

    # --- 4. Filter Data ---
    history_data = junction_data[
        (junction_data['DateTime'] <= selected_date) & 
        (junction_data['DateTime'] >= selected_date - pd.Timedelta(hours=72))
    ]
    future_data = junction_data[
        (junction_data['DateTime'] > selected_date) & 
        (junction_data['DateTime'] <= selected_date + pd.Timedelta(hours=24))
    ]
    future_data['Traffic_Level'] = future_data['Predicted_Vehicles'].apply(classify_traffic)

    # --- 5. Context Panel ---
    st.markdown("---")
    st.markdown(f"### 🧠 What the AI sees at **{selected_date.strftime('%I:%M %p on %b %d, %Y')}**")
    st.markdown("Before making a prediction, our model looks at the current environmental and historical context:")
    
    current_context = junction_data[junction_data['DateTime'] == selected_date]
    if not current_context.empty:
        ctx = current_context.iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡️ Temperature", f"{ctx['temperature']:.1f} °C")
        c2.metric("🌧️ Precipitation", f"{ctx['precipitation']:.1f} mm")
        c3.metric("🎉 Is it a Holiday?", "Yes (Traffic Behavior Changes)" if ctx['is_holiday'] == 1 else "No (Normal Day)")
        c4.metric("📈 Recent 6-Hour Trend", f"{ctx['rolling_mean_6h']:.1f} vehicles/hr")

    # --- 6. Chart ---
    st.markdown("### 🔮 The AI's 24-Hour Forecast")
    fig = build_traffic_chart(history_data, future_data, selected_date)
    st.plotly_chart(fig, use_container_width=True)

    # --- Google Maps Section ---
    st.markdown("---")
    st.markdown("### 🗺️ Live Junction Map")
    st.markdown("Colored markers show the **predicted traffic level** for each junction in the next hour.")
    
    api_key = st.secrets["GOOGLE_MAPS_API_KEY"]

    show_map_legend()
    build_google_map(future_data, selected_junction, api_key)

    # --- 7. Immediate Prediction ---
    if not future_data.empty:
        next_hour = future_data.iloc[0]
        st.markdown("### 🚨 Immediate Traffic Prediction")
        st.metric(
            label="Next Hour Traffic",
            value=classify_traffic(next_hour['Predicted_Vehicles'])
        )

    # --- 8. Next 3-Hour Predictions ---
    st.markdown("###🚦Traffic Prediction (Next 3 Hours)")
    future_preview = future_data.head(3)
    if future_preview.empty:
        st.info("No future data available for this selection.")
    else:
        for _,row in future_preview.iterrows():
            traffic_level = classify_traffic(row['Predicted_Vehicles'])
            time_str = row['DateTime'].strftime('%I:%M %p')

            if "Red" in traffic_level:
                st.error(f"{time_str} → {traffic_level}")
            elif "Yellow" in traffic_level:
                st.warning(f"{time_str} → {traffic_level}")
            else:
                st.success(f"{time_str} → {traffic_level}")

            st.write(
                f"{row['DateTime'].strftime('%I:%M %p')} → {classify_traffic(row['Predicted_Vehicles'])}"
            )

    # --- 9. Best Travel Time ---
    if not future_preview.empty:
        best_time = future_preview.loc[
            future_preview['Predicted_Vehicles'].idxmin()
        ]
        st.markdown("---")
        st.success(
            f"🚗 Best time to travel: **{best_time['DateTime'].strftime('%I:%M %p')}** "
            f"({classify_traffic(best_time['Predicted_Vehicles'])})"
        )

    # --- 10. Business Metrics ---
    st.markdown("---")
    st.markdown("### 🏆 Model Performance: How much better is this than a human guess?")
    st.info("To prove our AI is useful, we compared it against a 'Naive Human Guess' (assuming tomorrow's traffic will just be exactly the same as today's traffic).")
    
    improvement_pct = ((baseline_mae - ai_mae) / baseline_mae) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Human Guess Error Rate", value=f"Off by {baseline_mae:.1f} cars", help="On average, a simple human guess is wrong by this many vehicles per hour.")
    with col2:
        st.metric(label="AI Model Error Rate", value=f"Off by {ai_mae:.1f} cars", delta=f"AI is {improvement_pct:.1f}% more accurate", delta_color="inverse")
    with col3:
        j_mae = mean_absolute_error(junction_data['Vehicles'], junction_data['Predicted_Vehicles'])
        st.metric(label=f"Current Junction Accuracy", value=f"Off by {j_mae:.1f} cars")

    # --- 11. Technical Proof ---
    with st.expander("🛠️ Technical Details for Data Scientists (Cross-Validation)"):
        st.markdown("""
        **No Data Leakage & Robust Testing** We didn't just test this on one random subset of data. We used Scikit-Learn's `TimeSeriesSplit` to chronologically "walk forward" through the dataset across 4 different time periods. This ensures the model is actually learning the relationships between weather, holidays, and traffic, rather than just memorizing dates.
        """)
        cv_df = pd.DataFrame({
            "Validation Window": ["Fold 1 (Oldest Data)", "Fold 2", "Fold 3", "Fold 4 (Newest Data)"],
            "Average Error (MAE)": [f"{score:.2f} vehicles" for score in cv_scores]
        })
        st.table(cv_df)
        st.markdown("**Optimal Hyperparameters Found via RandomizedSearchCV:**")
        st.json(winning_params)

else:
    st.error("Not enough data.")

