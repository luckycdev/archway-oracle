def get_best_time_to_leave(future_df):
    """
    Scans the next 4 hours of predictions to find the lowest traffic volume.
    """
    if future_df.empty:
        return None

    # Look at only the next 4 hours from the current slider position
    next_4_hours = future_df.head(4)

    # Find the row with the minimum predicted volume
    best_row = next_4_hours.loc[next_4_hours['Predicted_Vehicles'].idxmin()]

    return {
        'time': best_row['DateTime'].strftime('%H:%M'),
        'volume': int(best_row['Predicted_Vehicles']),
        'reduction': int(future_df.iloc[0]['Predicted_Vehicles'] - best_row['Predicted_Vehicles'])
    }