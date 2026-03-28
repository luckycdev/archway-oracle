import pandas as pd

# Path to your heavy CSV
input_csv = "/workspaces/traffic-predictor/data/TrafficTab23.csv"
output_parquet = "/workspaces/traffic-predictor/data/TrafficTab23.parquet"

print("Reading CSV... this might take a minute...")
# We load it once. If this still crashes your RAM, add nrows=1000000
df = pd.read_csv(input_csv)

print("Converting 'timestamp' to datetime objects...")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Saving to Parquet format at {output_parquet}...")
# snappy compression is the standard for a balance of speed and size
df.to_parquet(output_parquet, compression='snappy', index=False)

print("Done! You can now delete the .csv to save space.")