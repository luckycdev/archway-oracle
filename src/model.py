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
    
    # CRITICAL: Convert road names to 'category' type
    df['road_segment_id'] = df['road_segment_id'].astype('category')
    
    # Sort by time for a valid forecast split
    df = df.sort_values('DateTime')
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # FIX: Put Categorical column FIRST so index 0 is the category
    X_train = train_df[CAT_FEATURES + FEATURES]
    y_train = train_df['vehicle_count']
    X_test = test_df[CAT_FEATURES + FEATURES]
    y_test = test_df['vehicle_count']
    
    # Define the mask: True for the first column (road_id), False for the rest (numeric)
    # Total columns = 1 (cat) + 11 (numeric) = 12
    cat_mask = [True] + [False] * len(FEATURES)

    # --- Model Setup ---
    model = HistGradientBoostingRegressor(
        categorical_features=cat_mask,
        random_state=42
    )
    
    # Simplified search grid for stability
    param_distributions = {
        'max_iter': [50, 100],
        'learning_rate': [0.1],
        'max_depth': [3, 5]
    }
    
    search = RandomizedSearchCV(
        model, 
        param_distributions, 
        n_iter=3, 
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_absolute_error',
        n_jobs=1
    )
    
    # This will now succeed because index 0 is correctly identified as a category
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # --- Evaluation ---
    predictions = best_model.predict(X_test)
    test_df = test_df.copy()
    test_df['Predicted_Vehicles'] = predictions
    
    ai_mae = mean_absolute_error(y_test, predictions)
    baseline_mae = mean_absolute_error(y_test, X_test['lag1'])
    
    from src.data_processing import classify_traffic
    test_df['Traffic_Level'] = test_df['Predicted_Vehicles'].apply(classify_traffic)
    
    return test_results_processing(test_df), ai_mae, baseline_mae, search.cv_results_, search.best_params_

def test_results_processing(df):
    # Ensure no negative predictions (AI can sometimes guess -5 cars)
    df['Predicted_Vehicles'] = df['Predicted_Vehicles'].clip(lower=0)
    return df