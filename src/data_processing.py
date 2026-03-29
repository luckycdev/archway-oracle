import pandas as pd
import numpy as np
import holidays

def classify_traffic(volume):
    if volume > 2000: return "🔴 Red (Heavy)"
    elif volume > 1000: return "🟡 Yellow (Moderate)"
    else: return "🟢 Green (Clear)"

def add_time_features(df):
    df['DateTime'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['DateTime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0).astype(np.float32)
    df['day'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = df['day'].isin([5, 6]).astype(np.int8)
    return df

def add_lag_features(df):
    df = df.sort_values(['road_segment_id', 'DateTime'])
    group = df.groupby('road_segment_id')['vehicle_count']
    df['lag1'] = group.shift(1).astype(np.float32)
    df['lag2'] = group.shift(2).astype(np.float32)
    df['rolling_mean_short'] = group.transform(lambda x: x.rolling(window=6).mean()).astype(np.float32)
    return df

def load_and_prep_data(filepath="data/stl_traffic_counts.csv"):
    # 1. Load the CSV
    df = pd.read_csv(filepath)
    
    # 2. Rename road segment ID for the model
    df = df.rename(columns={'Onstreet': 'road_segment_id'})

    # 3. Filter for recent data
    df = df[df['Year'] >= 2023].copy()

    # 4. FIX: Use 'x' and 'y' directly (from your CSV) for Coordinate Conversion
    # This math converts St. Louis State Plane coordinates to Lat/Long
    df['gps_latitude'] = 38.627 + (df['y'] - 1000059) / 364000
    df['gps_longitude'] = -90.199 + (df['x'] - 837675) / 288000
    
    # 5. Synthesize Hourly Data
    weights = {
        0: 0.012, 1: 0.008, 2: 0.005, 3: 0.005, 4: 0.01, 5: 0.03, 
        6: 0.06,  7: 0.085, 8: 0.08,  9: 0.05,  10: 0.045, 11: 0.05,
        12: 0.055, 13: 0.055, 14: 0.06, 15: 0.08, 16: 0.095, 17: 0.09,
        18: 0.06, 19: 0.04, 20: 0.03, 21: 0.025, 22: 0.02, 23: 0.015
    }
    
    hourly_dfs = []
    for hr, weight in weights.items():
        temp = df.copy()
        temp['hour'] = hr
        # Use 'AWT' (Average Weekly Traffic) from your CSV
        temp['vehicle_count'] = (temp['AWT'] * weight).astype(int)
        temp['timestamp'] = pd.Timestamp('2026-03-28') + pd.to_timedelta(hr, unit='h')
        hourly_dfs.append(temp)
    
    df = pd.concat(hourly_dfs).reset_index(drop=True)

    # 6. Pipeline
    df = add_time_features(df)
    
    # Add simple holiday check
    us_holidays = holidays.US(years=[2026])
    df['is_holiday'] = df['DateTime'].dt.date.isin(us_holidays).astype(np.int8)
    
    df = add_lag_features(df)
    df['weather_condition'] = 'Clear'
    
    return df.dropna()