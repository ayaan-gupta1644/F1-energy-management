import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Reading the .csv file
df = pd.read_csv('weather_data_suzuka_fullrace.csv')

# Reading the first few rows
print(df.head())

# Reading the columns
print(df.columns)

# Line graph for Air Temp and Track Temp
plt.figure(figsize=(12, 6))
plt.plot(df['LapNumber'], df['AirTemp'], label='Air Temperature')
plt.plot(df['LapNumber'], df['TrackTemp'], label='Track Temperature')

plt.xlabel('Lap')
plt.ylabel('Temperature (°C)')
plt.title('Air and Track Temperature Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Comparison between AirTemp and TrackTemp avg values
avg_values = {
    'AirTemp': df['AirTemp'].mean(),
    'TrackTemp': df['TrackTemp'].mean(),
}

plt.figure(figsize=(8, 5))
plt.bar(avg_values.keys(), avg_values.values())

plt.title('Average Temperature Values')
plt.ylabel('Average')
plt.show()

# Dual Axis Graph
lap_avg = (
    df.groupby("LapNumber")[["AirTemp", "TrackTemp"]]
    .mean()
    .reset_index()
)

# Create the figure and primary axis
fig, ax1 = plt.subplots(figsize=(14, 7))

# Left Y-axis: Air Temperature
ax1.plot(
    lap_avg["LapNumber"],
    lap_avg["AirTemp"],
    marker="o",
    linewidth=2,
    label="Air Temperature"
)
ax1.set_xlabel("Lap Number")
ax1.set_ylabel("Air Temperature (°C)")
ax1.set_xticks(lap_avg["LapNumber"])

# Right Y-axis: Track Temperature
ax2 = ax1.twinx()
ax2.plot(
    lap_avg["LapNumber"],
    lap_avg["TrackTemp"],
    marker="s",
    linewidth=2,
    linestyle="--",
    label="Track Temperature"
)
ax2.set_ylabel("Track Temperature (°C)")

# Title
plt.title("Lap-by-Lap Air Temperature vs Track Temperature")

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Grid and layout
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# Show plot
plt.show()
