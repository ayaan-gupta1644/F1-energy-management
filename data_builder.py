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

all_laps_data = []

print(f"Processing {len(laps)} laps for {driver}...")

#3. Processing & Gathering Telemetry Data
for i, lap in laps.iterlaps():
    # Get high-frequency telemetry
    tel = lap.get_telemetry().add_distance()
    
    # Calculate Basic Acceleration (Delta V / Delta T)
    tel['Basic Acc'] = tel['Speed'].diff() / tel['Time'].dt.total_seconds().diff()

    # Calculate Basic Brake Pressure % using Deceleration(-ve Acceleration)
    tel['Basic Brake_Pct'] = tel['Basic Acc'].apply(lambda x: abs(x) if x < 0 else 0)
    max_decel = tel['Basic Brake_Pct'].max()
    if max_decel > 0: 
        tel['Basic Brake_Pct'] = (tel['Basic Brake_Pct'] / max_decel) * 100
    
    # --- Add Context Metadata ---
    tel['LapNumber'] = lap['LapNumber']
    tel['Compound'] = lap['Compound']
    tel['TyreLife'] = lap['TyreLife']

    # Append to list
    all_laps_data.append(tel)

# 4. Combine and Export
final_df = pd.concat(all_laps_data, ignore_index=True)

# 5. Clean up NaN values created by the .diff() function
final_df.fillna(0, inplace=True)

final_df.to_csv(f'{driver}_suzuka_data_advanced.csv', index=False)
print("Dataset complete! Ready for model training.")