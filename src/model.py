import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from src.data_processing import classify_traffic

FEATURES = [
    'road_segment_id', 'gps_latitude', 'gps_longitude', 
    'hour', 'hour_sin', 'hour_cos', 
    'day', 'month', 'month_sin', 'month_cos', 'is_weekend', 
    'lag1', 'lag2', 'rolling_mean_short', 'rolling_mean_long', 
    'is_holiday', 'weather_condition'
]

TARGET = "vehicle_count"

CAT_COLS = ['road_segment_id', 'weather_condition']
categorical_mask = [True if f in CAT_COLS else False for f in FEATURES]

PARAM_DISTRIBUTIONS = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_depth': [3, 5, 10, 20, 30, 40, 50, None],
    'l2_regularization': [0.0, 0.1, 1.0]
}

def prepare_for_training(df):
    """Ensures categorical columns are formatted for HistGradientBoosting"""
    df = df.copy()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

def get_cat_mask():
    return [True if f in CAT_COLS else False for f in FEATURES]

def tune_and_train(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=3)

    cat_mask = get_cat_mask()
    model = HistGradientBoostingRegressor(random_state=42, categorical_features=cat_mask)

    search = RandomizedSearchCV(
        model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=5, 
        cv=tscv,   
        scoring='neg_mean_absolute_error',
        n_jobs=1,
        random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def run_cross_validation(df, best_params):
    """Evaluate the best params using TimeSeriesSplit cross-validation."""
    df = prepare_for_training(df)
    
    X_full = df[FEATURES]
    y_full = df[TARGET]
    tscv_eval = TimeSeriesSplit(n_splits=3)
    cv_scores = []

    cat_mask = get_cat_mask()

    for train_index, test_index in tscv_eval.split(X_full):
        cv_X_train, cv_X_test = X_full.iloc[train_index], X_full.iloc[test_index]
        cv_y_train, cv_y_test = y_full.iloc[train_index], y_full.iloc[test_index]
        
        cv_model = HistGradientBoostingRegressor(**best_params, categorical_features=cat_mask, random_state=42)
        cv_model.fit(cv_X_train, cv_y_train)
        
        preds = cv_model.predict(cv_X_test)
        cv_scores.append(mean_absolute_error(cv_y_test, preds))

    return cv_scores

def train_and_evaluate(df):
    """Full training pipeline: tune, train, evaluate, and return results."""
    df = df.sort_values('DateTime')
    df = prepare_for_training(df)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # Tune and train
    model, best_params = tune_and_train(X_train, y_train)

    # Evaluate on hold-out test set
    test_df['Predicted_Vehicles'] = model.predict(X_test)
    test_df['Traffic_Level'] = test_df['Predicted_Vehicles'].apply(classify_traffic)

    ai_mae = mean_absolute_error(y_test, test_df['Predicted_Vehicles'])
    baseline_mae = mean_absolute_error(y_test, test_df['lag1'])

    # Cross-validation
    cv_scores = run_cross_validation(df, best_params)

    return test_df, ai_mae, baseline_mae, cv_scores, best_params