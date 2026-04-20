import numpy as np
import pandas as pd
from config import (
    COMMUTE_GALLONS_PER_5_MIN,
    COMMUTE_MINUTES_PER_100_VEHICLES,
    GAS_PRICE_PER_GALLON,
    LOOKAHEAD_HOURS,
    WEATHER_SCORE_PENALTY_GLARE,
    WEATHER_SCORE_PENALTY_RAIN,
    WEATHER_SCORE_PENALTY_SNOW,
)

def get_best_time_to_leave(future_df, wait_penalty_per_hour=400):
    """
    Scans the next 4 hours of predictions to find the lowest traffic volume,
    incorporating time patience and weather risks.
    """
    if future_df.empty:
        return None

    # 1. Look at only the next 4 hours from the current slider position
    next_4_hours = future_df.head(LOOKAHEAD_HOURS).copy()

    # 2. Weather and visibility penalties
    # We add "virtual vehicles" to the score to discourage leaving during bad weather
    next_4_hours['Weather_Penalty'] = 0

    # If snowing, add a massive penalty (equivalent to 1000 extra cars)
    next_4_hours.loc[next_4_hours['is_snowing'] == 1, 'Weather_Penalty'] += WEATHER_SCORE_PENALTY_SNOW

    # If raining, add a moderate penalty
    next_4_hours.loc[next_4_hours['is_raining'] == 1, 'Weather_Penalty'] += WEATHER_SCORE_PENALTY_RAIN

    # If high glare, add a visibility penalty
    next_4_hours.loc[next_4_hours['sun_glare'] == 1, 'Weather_Penalty'] += WEATHER_SCORE_PENALTY_GLARE

    # 3. Calculate final adjusted score
    # Wait_Step represents hours waited (0, 1, 2, 3)
    next_4_hours['Wait_Step'] = np.arange(len(next_4_hours))

    # Score = Predicted Vehicles + Time Penalty + Weather Risk
    next_4_hours['Adjusted_Score'] = (
        next_4_hours['Predicted_Vehicles'] + 
        (next_4_hours['Wait_Step'] * wait_penalty_per_hour) + 
        next_4_hours['Weather_Penalty']
    )

    # 4. Find the row with the minimum adjusted score
    best_row = next_4_hours.loc[next_4_hours['Adjusted_Score'].idxmin()]

    return {
        'time': best_row['DateTime'].strftime('%I:%M %p'), # Using 12-hour format
        'volume': int(best_row['Predicted_Vehicles']),
        'reduction': int(future_df.iloc[0]['Predicted_Vehicles'] - best_row['Predicted_Vehicles']),
        'weather_hazard': best_row['Weather_Penalty'] > 0 
    }

def calculate_commute_impact(volume_reduction):
    """
    Estimates time and fuel savings based on vehicle reduction.
    Assumes a standard 15-mile St. Louis commute.
    """
    # Heuristic: For every 100 vehicles reduced, save ~2 mins of idling/stop-go
    minutes_saved = (volume_reduction / 100) * COMMUTE_MINUTES_PER_100_VEHICLES

    # Heuristic: Idling/Stop-go consumes ~0.05 gallons per 5 mins
    gallons_saved = (minutes_saved / 5) * COMMUTE_GALLONS_PER_5_MIN

    # Current STL Avg Gas Price (approx for 2026)
    fuel_money_saved = gallons_saved * GAS_PRICE_PER_GALLON

    return {
        'mins': round(max(0, minutes_saved), 1), # max(0,...) prevents negative savings
        'money': round(max(0, fuel_money_saved), 2)
    }