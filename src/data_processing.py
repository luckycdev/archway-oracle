import pandas as pd
import numpy as np
import holidays
import requests

# St. Louis Coordinates
LATITUDE = 38.6274
LONGITUDE = -90.1982

def classify_traffic(volume):
    if volume > 50: return "🔴 Red (Heavy)"
    elif volume > 20: return "🟡 Yellow (Moderate)"
    else: return "🟢 Green (Clear)"

def add_time_features(df):
    df['hour'] = df['DateTime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['day'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    df['is_weekend'] = df['day'].isin([5, 6]).astype(int)
    return df

def add_holiday_features(df):
    years = df['DateTime'].dt.year.unique().tolist()
    us_holidays = holidays.US(years=years)
    df['is_holiday'] = df['DateTime'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    return df

def add_lag_features(df):
    df = df.sort_values(['Junction', 'DateTime'])
    
    df['lag1'] = df.groupby('Junction')['Vehicles'].shift(1)
    df['lag2'] = df.groupby('Junction')['Vehicles'].shift(2)
    df['lag24'] = df.groupby('Junction')['Vehicles'].shift(24)
    
    df['rolling_mean_6h'] = df.groupby('Junction')['Vehicles'].transform(lambda x: x.rolling(window=6).mean())
    df['rolling_mean_12h'] = df.groupby('Junction')['Vehicles'].transform(lambda x: x.rolling(window=12).mean())
    return df

def fetch_weather(df):
    start_date = df['DateTime'].min().strftime('%Y-%m-%d')
    end_date = df['DateTime'].max().strftime('%Y-%m-%d')
    
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

    return df

def load_and_prep_data(filepath="/workspaces/traffic-predictor/data/traffic.csv"):
    df = pd.read_csv(filepath)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = add_time_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df)
    df = fetch_weather(df)
    df = df.dropna()
    return df