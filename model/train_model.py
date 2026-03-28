import pandas as pd 

# Load dataset
df = pd.read_csv("traffic.csv")

# Convert DateTime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract time features
df['hour'] = df['DateTime'].dt.hour
df['day'] = df['DateTime'].dt.dayofweek
df['month'] = df['DateTime'].dt.month
df['is_weekend'] = df['day'].isin([5, 6]).astype(int)

# Sort data
df = df.sort_values(['Junction', 'DateTime'])

# Create lag features
df['lag1'] = df.groupby('Junction')['Vehicles'].shift(1)
df['lag2'] = df.groupby('Junction')['Vehicles'].shift(2)

# Drop missing values (from lagging)
df = df.dropna()

# Train model with scikit-learn
