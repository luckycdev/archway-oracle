import pandas as pd
import numpy as np
import holidays

def classify_traffic(volume):
    if volume > 80: return "🔴 Red (Heavy)"
    elif volume > 40: return "🟡 Yellow (Moderate)"
    else: return "🟢 Green (Clear)"

def add_time_features(df):
    df['DateTime'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['DateTime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0).astype(np.float32)
    df['day'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0).astype(np.float32)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0).astype(np.float32)
    df['is_weekend'] = df['day'].isin([5, 6]).astype(np.int8)
    return df

def add_holiday_features(df):
    years = df['DateTime'].dt.year.unique().tolist()
    us_holidays = holidays.US(years=years)
    holiday_set = set(us_holidays.keys()) # Using a set lookup is much faster for millions of rows
    df['is_holiday'] = df['DateTime'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    return df

def add_lag_features(df):
    df = df.sort_values(['road_segment_id', 'DateTime'])
    
    group = df.groupby('road_segment_id')['vehicle_count']

    df['lag1'] = group.shift(1)
    df['lag2'] = group.shift(2)

    df['rolling_mean_short'] = group.transform(lambda x: x.rolling(window=6).mean())
    df['rolling_mean_long'] = group.transform(lambda x: x.rolling(window=12).mean())

    return df

def load_and_prep_data(filepath="/workspaces/traffic-predictor/data/TrafficTab23.parquet"):
    cols_to_load = [
        'timestamp', 'road_segment_id', 'vehicle_count',
        'gps_latitude', 'gps_longitude', 'weather_condition',
        'road_surface_status'
    ]

    # Fast binary load
    df = pd.read_parquet(filepath, columns=cols_to_load)

    # 1. Preprocessing Pipeline
    df = add_time_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df)

    # 2. Impute missing weather
    if 'weather_condition' in df.columns:
        df['weather_condition'] = df['weather_condition'].fillna('Clear')

    # 3. Clean up
    df = df.dropna()

    # Downcast to save 50% RAM on these columns
    df['gps_latitude'] = df['gps_latitude'].astype(np.float32)
    df['gps_longitude'] = df['gps_longitude'].astype(np.float32)
    
    return df