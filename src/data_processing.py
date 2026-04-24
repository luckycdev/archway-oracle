import pandas as pd
import numpy as np
import holidays
import requests
from pathlib import Path
from datetime import datetime, timedelta
from config import (
    DATA_FILE_PATH,
    DATA_MIN_YEAR,
    GPS_LAT_BASE,
    GPS_LAT_SCALE,
    GPS_LON_BASE,
    GPS_LON_SCALE,
    HOLIDAY_YEARS,
    MORNING_SIM_PRECIPITATION,
    MORNING_SIM_TEMPERATURE,
    MORNING_WEATHER_END_HOUR,
    MORNING_WEATHER_START_HOUR,
    OPEN_METEO_BASE_URL,
    OPEN_METEO_LATITUDE,
    OPEN_METEO_LONGITUDE,
    OPEN_METEO_TIMEOUT_SECONDS,
    SUN_GLARE_CLOUD_THRESHOLD,
    SUN_GLARE_HOURS,
    SYNTH_DAYS_BACK,
    SYNTH_DAYS_FORWARD,
    SYNTH_NOISE_MAX,
    SYNTH_NOISE_MIN,
    WEATHER_FALLBACK_CLOUD_COVER,
    WEATHER_FALLBACK_PRECIPITATION,
    WEATHER_FALLBACK_TEMPERATURE,
)

# Monthly Expansion Factors (Standard Traffic Engineering Estimates)
MONTHLY_FACTORS = {
    1: 0.90, 2: 0.92, 3: 0.98, 4: 1.02, 5: 1.05, 6: 1.10,
    7: 1.12, 8: 1.10, 9: 1.03, 10: 1.01, 11: 0.95, 12: 0.93,
}


def calculate_daily_traffic(df, segment_id, target_date):
    """Sums vehicle count for 24 hours of selected day."""
    if hasattr(target_date, "date"):
        target_date_only = target_date.date()
    else:
        target_date_only = target_date

    day_data = df[
        (df["road_segment_id"] == segment_id)
        & (df["DateTime"].dt.date == target_date_only)
    ]
    return int(day_data["vehicle_count"].sum())


def calculate_aadt_estimate(daily_total, target_date):
    """Estimates Annual Average Daily Traffic from daily volume."""
    month = target_date.month
    factor = MONTHLY_FACTORS.get(month, 1.0)
    return int(daily_total / factor)

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


def _resolve_data_path(filepath):
    candidate = Path(filepath)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    if candidate.exists():
        return candidate

    module_relative = Path(__file__).resolve().parent / candidate
    if module_relative.exists():
        return module_relative

    return candidate

def load_and_prep_data(filepath=DATA_FILE_PATH):
    # 1. Load the CSV
    df = pd.read_csv(_resolve_data_path(filepath))
    
    # 2. Rename road segment ID for the model
    df = df.rename(columns={'Onstreet': 'road_segment_id'})

    # 3. Filter for recent data
    df = df[df['Year'] >= DATA_MIN_YEAR].copy()

    # 4. Accurate Coordinate Conversion (State Plane Feet to Lat/Long)
    # Calibrated for St. Louis County (Des Peres, Kirkwood, Mehlville area)
    df['gps_latitude'] = GPS_LAT_BASE + (df['y'] * GPS_LAT_SCALE)
    df['gps_longitude'] = GPS_LON_BASE + (df['x'] * GPS_LON_SCALE)
    
    # 5. Synthesize Hourly Data
    weights = {
        0: 0.012, 1: 0.008, 2: 0.005, 3: 0.005, 4: 0.01, 5: 0.03, 
        6: 0.06,  7: 0.085, 8: 0.08,  9: 0.05,  10: 0.045, 11: 0.05,
        12: 0.055, 13: 0.055, 14: 0.06, 15: 0.08, 16: 0.095, 17: 0.09,
        18: 0.06, 19: 0.04, 20: 0.03, 21: 0.025, 22: 0.02, 23: 0.015
    }
    
    today_date = datetime.now().date()
    dates_to_generate = [
        today_date + timedelta(days=offset)
        for offset in range(-SYNTH_DAYS_BACK, SYNTH_DAYS_FORWARD + 1)
    ]
    
    hourly_dfs = []
    for d in dates_to_generate: # Loop through all 3 days
        for hr, weight in weights.items():
            temp = df.copy()
            temp['hour'] = hr
            base_volume = temp['AWT'] * weight
            noise_multiplier = np.random.uniform(SYNTH_NOISE_MIN, SYNTH_NOISE_MAX, size=len(temp))
            temp['vehicle_count'] = (base_volume * noise_multiplier).astype(int)
            temp['timestamp'] = pd.Timestamp(d) + pd.to_timedelta(hr, unit='h')
            hourly_dfs.append(temp)
    
    df = pd.concat(hourly_dfs).reset_index(drop=True)

    # 6. Weather engine
    # Default values
    df = add_time_features(df)
    df['temperature'] = 22.0
    df['precipitation'] = 0.0

    # Simulate a cold, rainy/snowy morning (6A M - 10 AM)
    # This gives the AI something to "learn" from
    morning_mask = (df['hour'] >= MORNING_WEATHER_START_HOUR) & (df['hour'] <= MORNING_WEATHER_END_HOUR)
    df.loc[morning_mask, 'precipitation'] = MORNING_SIM_PRECIPITATION
    df.loc[morning_mask, 'temperature'] = MORNING_SIM_TEMPERATURE
    
    start_date = df['DateTime'].min().strftime('%Y-%m-%d')
    end_date = df['DateTime'].max().strftime('%Y-%m-%d')
    
    # 6. Weather engine (API Integration)
    LATITUDE = OPEN_METEO_LATITUDE
    LONGITUDE = OPEN_METEO_LONGITUDE
    
    start_date = dates_to_generate[0].strftime('%Y-%m-%d')
    end_date = dates_to_generate[-1].strftime('%Y-%m-%d')
    
    weather_url = (
        f"{OPEN_METEO_BASE_URL}?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,precipitation,cloud_cover&timezone=auto" # Added cloud_cover to the request
    )
    
    try:
        response = requests.get(weather_url, timeout=OPEN_METEO_TIMEOUT_SECONDS)
        if response.status_code == 200:
            data = response.json()
            weather_df = pd.DataFrame({
                'DateTime': pd.to_datetime(data['hourly']['time']),
                'temp_api': data['hourly']['temperature_2m'],
                'precip_api': data['hourly']['precipitation'],
                'cloud_api': data['hourly']['cloud_cover']
            })
            # Merge and use the API values
            df = pd.merge(df, weather_df, on='DateTime', how='left')
            df['temperature'] = df['temp_api'].fillna(WEATHER_FALLBACK_TEMPERATURE)
            df['precipitation'] = df['precip_api'].fillna(WEATHER_FALLBACK_PRECIPITATION)
            df['cloud_cover'] = df['cloud_api'].fillna(WEATHER_FALLBACK_CLOUD_COVER)
            # Drop the temporary helper columns
            df = df.drop(columns=['temp_api', 'precip_api', 'cloud_api'])
        else:
            raise ValueError("API Offline")
    except Exception:
        # Robust Fallbacks
        df['temperature'] = WEATHER_FALLBACK_TEMPERATURE
        df['precipitation'] = WEATHER_FALLBACK_PRECIPITATION
        df['cloud_cover'] = WEATHER_FALLBACK_CLOUD_COVER

    # Logic for Rain/Snow/Glare (Apply this AFTER the merge)
    df['is_snowing'] = ((df['precipitation'] > 0) & (df['temperature'] <= 0)).astype(np.int8)
    df['is_raining'] = ((df['precipitation'] > 0) & (df['temperature'] > 0)).astype(np.int8)
    
    df['sun_glare'] = ((df['cloud_cover'] < SUN_GLARE_CLOUD_THRESHOLD) & (df['hour'].isin(SUN_GLARE_HOURS))).astype(np.int8)
    
    # Add simple holiday check
    us_holidays = holidays.US(years=HOLIDAY_YEARS)
    df['is_holiday'] = df['DateTime'].dt.date.isin(us_holidays).astype(np.int8)
    
    df = add_lag_features(df)
    df['weather_condition'] = 'Clear'
    
    return df.dropna()