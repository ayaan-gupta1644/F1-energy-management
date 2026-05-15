import fastf1
import pandas as pd

# 1. Setup Cache (Crucial for large datasets)
fastf1.Cache.enable_cache('f1_cache') 

# 2. Load the 2025 Suzuka Race
# 'Suzuka' or the round number can be used
session = fastf1.get_session(2025, 'Suzuka', 'R')
session.load()

# 3. Get all laps for Max Verstappen
ver_laps = session.laps.pick_driver('VER')

# 4. Loop through all laps and collect telemetry
all_telemetry = pd.DataFrame()

for i, lap in ver_laps.iterlaps():
    # Get telemetry for this specific lap
    # .add_distance() is vital for comparing different laps on the same track points
    tel = lap.get_telemetry().add_distance()
    
    # Add LapNumber so your AI knows which stage of the race it is
    tel['LapNumber'] = lap['LapNumber']
    
    # Add Tire Compound (Strategy affects energy usage)
    tel['Compound'] = lap['Compound']
    
    # Append to our master dataframe
    all_telemetry = pd.concat([all_telemetry, tel], ignore_index=True)

# 5. Save for your AI model
all_telemetry.to_csv('ver_suzuka_2025_full_race.csv', index=False)
print("Data Exported! Ready for training.")