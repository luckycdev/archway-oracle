import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# 1. Define features - Road ID must be handled separately
FEATURES = [
    'gps_latitude', 'gps_longitude', 
    'hour', 'hour_sin', 'hour_cos', 
    'day', 'is_weekend', 'is_holiday',
    'lag1', 'lag2', 'rolling_mean_short'
]
CAT_FEATURES = ['road_segment_id']

def train_and_evaluate(df):
    df = df.copy()
    df['road_segment_id'] = df['road_segment_id'].astype('category')
    
    # Sort and Split
    df = df.sort_values('DateTime')
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[CAT_FEATURES + FEATURES]
    y_train = train_df['vehicle_count']
    X_test = test_df[CAT_FEATURES + FEATURES]
    y_test = test_df['vehicle_count']
    
    # Train Model
    cat_mask = [True] + [False] * len(FEATURES)
    model = HistGradientBoostingRegressor(categorical_features=cat_mask, random_state=42)
    model.fit(X_train, y_train)
    
    # --- FIX: Predict for the WHOLE day so the map works 24/7 ---
    X_all = df[CAT_FEATURES + FEATURES]
    df['Predicted_Vehicles'] = model.predict(X_all).clip(min=0)
    
    # Calculate metrics on the unseen test set for accuracy reporting
    test_preds = model.predict(X_test)
    ai_mae = mean_absolute_error(y_test, test_preds)
    baseline_mae = mean_absolute_error(y_test, X_test['lag1'])
    
    from src.data_processing import classify_traffic
    df['Traffic_Level'] = df['Predicted_Vehicles'].apply(classify_traffic)
    
    # Return the FULL dataframe so the map/slider work for all hours
    return df, ai_mae, baseline_mae, {}, {}

def test_results_processing(df):
    # Ensure no negative predictions (AI can sometimes guess -5 cars)
    df['Predicted_Vehicles'] = df['Predicted_Vehicles'].clip(lower=0)
    return df