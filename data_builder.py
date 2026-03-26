import fastf1
import pandas as pd
import numpy as np

# 1. Setup
fastf1.Cache.enable_cache('f1_cache') 
session = fastf1.get_session(2025, 'Suzuka', 'R')
session.load()

# 2. Select Driver
driver = 'VER'
laps = session.laps.pick_driver(driver)

# 3. Get Weather (Resample to match race length)
weather_data = session.weather_data

all_laps_data = []

print(f"Processing {len(laps)} laps for {driver}...")

for i, lap in laps.iterlaps():
    # Get high-frequency telemetry
    tel = lap.get_telemetry().add_distance()

    # Merge Weather (Closest timestamp)
    tel['TrackTemp'] = weather_data.iloc[0]['TrackTemp'] # Simplification
    
    # --- Feature Engineering for AI ---
    # Calculate Acceleration (Delta V / Delta T)
    tel['Acc'] = tel['Speed'].diff() / tel['Time'].dt.total_seconds().diff()
    
    # --- Add Context Metadata ---
    tel['LapNumber'] = lap['LapNumber']
    tel['Compound'] = lap['Compound']
    tel['TyreLife'] = lap['TyreLife']
    
    # 1. Handle Brake (Check if it's already % or needs a proxy)
    if tel['Brake'].max() > 1:
        # Data is already providing a percentage/pressure
        tel['Brake_Pct'] = tel['Brake']
    else:
        # Data is Boolean; create a proxy based on deceleration (Acc < 0)
        # We normalize negative acceleration to a 0-100 scale
        tel['Brake_Pct'] = tel['Acc'].apply(lambda x: abs(x) if x < 0 else 0)
        max_decel = tel['Brake_Pct'].max()
        if max_decel > 0:
            tel['Brake_Pct'] = (tel['Brake_Pct'] / max_decel) * 100

    # 2. Robust Harvesting Indicator
    # Harvesting = Braking AND Speed > 100 km/h (MGU-K needs rotation to work)
    tel['Is_Harvesting'] = ((tel['Brake_Pct'] > 5) & (tel['Speed'] > 100)).astype(int)

    # Append to list
    all_laps_data.append(tel)

# 4. Combine and Export
final_df = pd.concat(all_laps_data, ignore_index=True)

# Clean up NaN values created by the .diff() function
final_df.fillna(0, inplace=True)

final_df.to_csv(f'{driver}_suzuka_data_advanced.csv', index=False)
print("Dataset complete! Ready for model training.")