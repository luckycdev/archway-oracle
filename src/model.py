import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

# Define features
FEATURES = [
    'gps_latitude', 'gps_longitude', 
    'hour', 'hour_sin', 'hour_cos', 
    'day', 'is_weekend', 'is_holiday',
    'lag1', 'lag2', 'rolling_mean_short'
]
# Added safety: we'll check if these weather columns exist in df
WEATHER_FEATURES = ['temperature', 'precipitation', 'cloud_cover', 'sun_glare']

def train_and_evaluate(df):
    df = df.copy()
    
    # 1. Setup Categorical Data
    df['road_segment_id'] = df['road_segment_id'].astype('category')
    
    # Ensure all expected columns exist (fills with 0 if missing)
    for col in FEATURES + WEATHER_FEATURES:
        if col not in df.columns:
            df[col] = 0

    actual_features = FEATURES + [f for f in WEATHER_FEATURES if f in df.columns]
    
    # 2. Sort and Time-Series Split
    df = df.sort_values('DateTime')
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[['road_segment_id'] + actual_features]
    y_train = train_df['vehicle_count']
    X_test = test_df[['road_segment_id'] + actual_features]
    y_test = test_df['vehicle_count']
    
    # 3. Train Model
    # categorical_features=[0] tells the model the first column is the Road ID category
    model = HistGradientBoostingRegressor(categorical_features=[0], random_state=42, max_iter=100)
    model.fit(X_train, y_train)
    
    # 4. Calculate Feature Importance
    # HistGradientBoosting doesn't have .feature_importances_, so we use permutation_importance
    result = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42)
    fi_dict = dict(zip(X_test.columns, result.importances_mean))
    
    # 5. Generate Predictions for the ENTIRE dataset
    X_all = df[['road_segment_id'] + actual_features]
    df['Predicted_Vehicles'] = model.predict(X_all).clip(min=0)
    
    # 6. Metrics
    test_preds = model.predict(X_test)
    ai_mae = mean_absolute_error(y_test, test_preds)
    # Baseline: "Tomorrow will be just like today" (using lag1)
    baseline_mae = mean_absolute_error(y_test, X_test['lag1'])
    
    # 7. Classification (Sync with data_processing.py)
    from src.data_processing import classify_traffic
    df['Traffic_Level'] = df['Predicted_Vehicles'].apply(classify_traffic)
    
    # Define placeholders for things not currently calculated to prevent Unpack errors
    cv_scores = {} 
    winning_params = model.get_params()

    # RETURN: Exactly 6 items in the order app.py expects
    return df, ai_mae, baseline_mae, cv_scores, winning_params, fi_dict