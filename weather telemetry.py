import fastf1 
import pandas as pd 
import numpy as np 

fastf1.Cache.enable_cache('f1_cache') 
session = fastf1.get_session(2025, 'Suzuka', 'R') 
session.load()

weather_data = session.weather_data

laps = session.laps.pick_driver('VER')
all_laps_data = []

print(f"Processing {len(laps)} laps weather data...") 

for i, lap in laps.iterlaps():
    tel = lap.get_telemetry().add_distance()
    lap_start = lap['LapStartTime'] 

    # Add lap metadata to every telemetry row
    tel['LapNumber'] = lap['LapNumber']

    idx = (weather_data['Time'] - lap_start).abs().idxmin() 
    weather_row = weather_data.loc[idx]

    # Assign weather to all telemetry rows in this lap
    tel['TrackTemp'] = weather_row['TrackTemp']
    tel['AirTemp'] = weather_row['AirTemp']
    tel['Pressure'] = weather_row['Pressure']
    tel['Humidity'] = weather_row['Humidity']
    tel['Rainfall'] = weather_row['Rainfall']
    tel['WindSpeed'] = weather_row['WindSpeed']
    tel['WindDirection'] = weather_row['WindDirection'] 
    
    # Append to list 
    all_laps_data.append(tel) 

#Combine and Export 
final_df = pd.concat(all_laps_data, ignore_index=True) 
    
# 5. Clean up NaN values created by the .diff() function 
final_df.fillna(0, inplace=True) 
    
# Keep only relevant columns
relevant_columns = [
    'LapNumber',
    'TrackTemp',
    'AirTemp',
    'Pressure',
    'Humidity',
    'Rainfall',
    'WindSpeed',
    'WindDirection'
]

# Select only columns that actually exist
final_df = final_df[[col for col in relevant_columns if col in final_df.columns]]

# Export to CSV
final_df.to_csv('weather_data_suzuka_fullrace.csv', index=False)

print("Dataset complete! Ready for model training.")
