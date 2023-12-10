import pandas as pd
from prophet import Prophet
import pickle
import os
import re

def safe_filename(name):
    return re.sub(r'\W+', '_', name)

# Read the CSV file
df = pd.read_csv('citibike.csv', low_memory=False)

# Filter out rows where the start and end stations are the same
df = df[df['start_station_name'] != df['end_station_name']]

# Convert start and end times to datetime
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])

# Create separate dataframes for starts and ends, renaming columns to a common format
df_starts = df[['started_at', 'start_station_name']].rename(columns={'started_at': 'timestamp', 'start_station_name': 'station'})
df_ends = df[['ended_at', 'end_station_name']].rename(columns={'ended_at': 'timestamp', 'end_station_name': 'station'})

# Concatenate the two dataframes
df_combined = pd.concat([df_starts, df_ends])

# Group by station and hour, counting the number of rides
df_grouped = df_combined.groupby(['station', df_combined['timestamp'].dt.floor('H')]).size().reset_index(name='count')

# Directory to save models
model_dir = 'prophet_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Iterate over each station
for selected_station in df_grouped['station'].unique():
    print(f"Training model for {selected_station}")
    df_station = df_grouped[df_grouped['station'] == selected_station].copy()
    df_station.rename(columns={'timestamp': 'ds', 'count': 'y'}, inplace=True)

    try:
        model = Prophet()
        model.fit(df_station)

        model_file_name = f'prophet_model_{safe_filename(selected_station)}.pkl'
        with open(os.path.join(model_dir, model_file_name), 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Error training model for station {selected_station}: {e}")
