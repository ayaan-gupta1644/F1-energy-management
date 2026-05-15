import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('weather_data_suzuka_fullrace.csv')

# Convert Rainfall column from True/False strings to boolean
df["Rainfall"] = (
    df["Rainfall"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"true": True, "false": False})
)

# Average humidity for each lap
# For Rainfall, use .any() so that if it rained at any point during the lap,
# the entire lap is marked as rainy.
lap_weather = (
    df.groupby("LapNumber")
    .agg({
        "Humidity": "mean",
        "Rainfall": "any"
    })
    .reset_index()
)

# Separate rainy and dry laps
rain_laps = lap_weather[lap_weather["Rainfall"] == True]
dry_laps = lap_weather[lap_weather["Rainfall"] == False]

# Create figure
plt.figure(figsize=(14, 7))

# Plot humidity line
plt.plot(
    lap_weather["LapNumber"],
    lap_weather["Humidity"],
    linewidth=2,
    marker="o",
    label="Humidity (%)"
)

# Blue markers for rainy laps
plt.scatter(
    rain_laps["LapNumber"],
    rain_laps["Humidity"],
    s=120,
    marker="o",
    label="Rain"
)

# Gray markers for dry laps
plt.scatter(
    dry_laps["LapNumber"],
    dry_laps["Humidity"],
    s=60,
    marker="o",
    alpha=0.5,
    label="No Rain"
)

# Labels and title
plt.xlabel("Lap Number")
plt.ylabel("Humidity (%)")
plt.title("Lap-by-Lap Humidity with Rain Markers")

# Show every lap number on the x-axis
plt.xticks(lap_weather["LapNumber"])

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
