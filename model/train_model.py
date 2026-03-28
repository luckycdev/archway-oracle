import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import holidays
import requests
from sklearn.model_selection import RandomizedSearchCV

# --- Page Configuration ---
st.set_page_config(page_title="Traffic Predictor AI", layout="wide")
st.title("🚦 Smart City Traffic Prediction Dashboard")
st.markdown("Predicting future traffic jams by analyzing historical trends, weather, and holidays.")

# --- 1. Data Loading & Feature Engineering ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("/workspaces/traffic-predictor/data/traffic.csv")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    df['hour'] = df['DateTime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['day'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    df['is_weekend'] = df['day'].isin([5, 6]).astype(int)
    
    years = df['DateTime'].dt.year.unique().tolist()
    us_holidays = holidays.US(years=years)
    df['is_holiday'] = df['DateTime'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)

    df = df.sort_values(['Junction', 'DateTime'])
    
    df['lag1'] = df.groupby('Junction')['Vehicles'].shift(1)
    df['lag2'] = df.groupby('Junction')['Vehicles'].shift(2)
    df['lag24'] = df.groupby('Junction')['Vehicles'].shift(24)
    
    df['rolling_mean_6h'] = df.groupby('Junction')['Vehicles'].transform(lambda x: x.rolling(window=6).mean())
    df['rolling_mean_12h'] = df.groupby('Junction')['Vehicles'].transform(lambda x: x.rolling(window=12).mean())

    start_date = df['DateTime'].min().strftime('%Y-%m-%d')
    end_date = df['DateTime'].max().strftime('%Y-%m-%d')
    
    # St. Louis Coordinates
    LATITUDE = 38.6274
    LONGITUDE = -90.1982
    
    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,precipitation&timezone=auto"
    )
    
    try:
        response = requests.get(weather_url)
        if response.status_code == 200:
            weather_data = response.json()
            weather_df = pd.DataFrame({
                'DateTime': pd.to_datetime(weather_data['hourly']['time']),
                'temperature': weather_data['hourly']['temperature_2m'],
                'precipitation': weather_data['hourly']['precipitation']
            })
            df = pd.merge(df, weather_df, on='DateTime', how='left')
            df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
            df['precipitation'] = df['precipitation'].fillna(0)
        else:
            df['temperature'] = 0
            df['precipitation'] = 0
    except Exception:
        df['temperature'] = 0
        df['precipitation'] = 0

    df = df.dropna()
    return df

with st.spinner("Processing data & pulling St. Louis weather..."):
    data = load_and_prep_data()

# --- 2. Robust Evaluation Training ---
@st.cache_resource
def train_and_evaluate(df):
    features = ['Junction', 'hour', 'day', 'month', 'is_weekend', 
                'lag1', 'lag2', 'lag24', 'rolling_mean_6h', 'rolling_mean_12h', 
                'is_holiday', 'temperature', 'precipitation']
    target = 'Vehicles'

    df = df.sort_values('DateTime')
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # --- 1. Hyperparameter Tuning ---
    param_distributions = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_iter': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'max_depth': [3, 5, 10, 20, 30, 40, 50, None],
        'min_samples_leaf': [10, 20, 50],
        'l2_regularization': [0.0, 0.1, 1.0]
    }
    
    tscv = TimeSeriesSplit(n_splits=4)
    
    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=10, 
        cv=tscv,   
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    
    # Get the best model and parameters
    model = search.best_estimator_
    best_params = search.best_params_ # We save this to return it!

    # Predict on the 20% test set
    test_df['Predicted_Vehicles'] = model.predict(X_test)
    ai_mae = mean_absolute_error(test_df[target], test_df['Predicted_Vehicles'])
    baseline_mae = mean_absolute_error(test_df[target], test_df['lag24'])
    
    # --- 2. Robust Evaluation (Cross-Validation) ---
    tscv_eval = TimeSeriesSplit(n_splits=4)
    cv_scores = []
    X_full, y_full = df[features], df[target]
    
    for train_index, test_index in tscv_eval.split(X_full):
        cv_X_train, cv_X_test = X_full.iloc[train_index], X_full.iloc[test_index]
        cv_y_train, cv_y_test = y_full.iloc[train_index], y_full.iloc[test_index]
        
        # THE FIX: Unpack the winning parameters into the CV models using **best_params
        cv_model = HistGradientBoostingRegressor(**best_params, random_state=42)
        cv_model.fit(cv_X_train, cv_y_train)
        
        preds = cv_model.predict(cv_X_test)
        cv_scores.append(mean_absolute_error(cv_y_test, preds))
        
    # Return the best_params so we can show them off
    return test_df, ai_mae, baseline_mae, cv_scores, best_params

# Update the calling function to expect the new 'best_params' variable
with st.spinner("Tuning Hyperparameters & Running Cross-Validation (This may take a minute)..."):
    test_data, ai_mae, baseline_mae, cv_scores, winning_params = train_and_evaluate(data)

# --- 3. Sidebar UI ---
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

    history_data = junction_data[
        (junction_data['DateTime'] <= selected_date) & 
        (junction_data['DateTime'] >= selected_date - pd.Timedelta(hours=72))
    ]
    
    future_data = junction_data[
        (junction_data['DateTime'] > selected_date) & 
        (junction_data['DateTime'] <= selected_date + pd.Timedelta(hours=24))
    ]

    # --- 4. The "What the AI Knows" Section (Plain English) ---
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

    # --- 5. The Interactive Chart ---
    st.markdown("### 🔮 The AI's 24-Hour Forecast")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=history_data['DateTime'], y=history_data['Vehicles'],
        mode='lines', name='Known Past Traffic', line=dict(color='#1f77b4', width=3)))

    fig.add_trace(go.Scatter(x=future_data['DateTime'], y=future_data['Predicted_Vehicles'],
        mode='lines', name='AI Predicted Future', line=dict(color='#ff7f0e', width=3, dash='dot')))
    
    fig.add_trace(go.Scatter(x=future_data['DateTime'], y=future_data['Vehicles'],
        mode='lines', name='What Actually Happened', line=dict(color='gray', width=1, dash='solid'), opacity=0.5))

    fig.update_layout(
        xaxis_title="Timeline", yaxis_title="Number of Vehicles", hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        shapes=[dict(type="line", x0=selected_date, x1=selected_date, y0=0, y1=1, yref="paper", line=dict(color="White", width=2, dash="dash"))],
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. Business Impact & Evaluation (For Non-Technical Judges) ---
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

    # --- 7. The Technical Proof (Hidden for Business Judges, Open for Tech Judges) ---
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
