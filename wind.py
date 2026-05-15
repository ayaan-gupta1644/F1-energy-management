import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("weather_data_suzuka_fullrace.csv")

# Aggregate wind data lap-by-lap
lap_wind = (
    df.groupby("LapNumber")
    .agg({
        "WindSpeed": "mean",
        "WindDirection": "mean"
    })
    .reset_index()
)

# Extract data
laps = lap_wind["LapNumber"].to_numpy()
wind_speed = lap_wind["WindSpeed"].to_numpy()
wind_direction = lap_wind["WindDirection"].to_numpy()

# Convert direction to radians for polar plotting
theta = np.radians(wind_direction)

# Normalize speeds for color mapping
norm = plt.Normalize(wind_speed.min(), wind_speed.max())
colors = plt.cm.viridis(norm(wind_speed))

# Create polar plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

# Plot bars
bars = ax.bar(
    theta,                  # direction (angle)
    wind_speed,             # speed (radius)
    width=np.radians(6),    # bar width in degrees
    bottom=0,
    color=colors,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.85
)

# Set meteorological orientation:
# 0° = North, angles increase clockwise
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# Direction labels
ax.set_thetagrids(
    range(0, 360, 45),
    labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
)

# Title
ax.set_title(
    "Wind Direction and Speed (Lap-by-Lap)",
    fontsize=16,
    y=1.06,
    pad=10
)

# Grid styling
ax.grid(alpha=0.3)

# Add colorbar for wind speed
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.10, shrink=0.8)
cbar.set_label("Wind Speed (m/s)")

for i, lap in enumerate(lap_wind["LapNumber"]):
    if lap % 5 == 0:
        ax.text(
            theta[i],
            wind_speed[i] + 0.3,
            str(int(lap)),
            ha="center",
            va="center",
            fontsize=8
        )

plt.subplots_adjust(top=0.88, right=0.88)
plt.show()



# Creating a polar plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

# Scatter plot:
# angle  = wind direction
# radius = wind speed
# color  = lap number
# size   = wind speed (larger points for stronger wind)
scatter = ax.scatter(
    theta,
    wind_speed,
    c=laps,
    s=wind_speed * 30,   # Adjust multiplier to change marker size
    cmap="viridis",
    alpha=0.8,
    edgecolors="black",
    linewidth=0.5
)

# Set meteorological orientation
# 0° = North, angles increase clockwise
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# Compass labels
ax.set_thetagrids(
    range(0, 360, 45),
    labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
)

# Radial label
ax.set_ylabel("Wind Speed (m/s)", labelpad=30)

# Title
ax.set_title(
    "Lap-by-Lap Polar Scatter Plot of Wind Speed and Direction",
    fontsize=16,
    y=1.06
)

# Grid styling
ax.grid(alpha=0.3)

# Colorbar showing lap number
cbar = fig.colorbar(scatter, ax=ax, pad=0.10, shrink=0.8)
cbar.set_label("Lap Number")

# Annotate every 5th lap to reduce clutter
for i, lap in enumerate(laps):
    if lap % 5 == 0:
        ax.text(
            theta[i],
            wind_speed[i] + 0.15,
            str(int(lap)),
            fontsize=8,
            ha="center",
            va="center"
        )

# Ensure title remains visible in fullscreen
plt.subplots_adjust(top=0.88, right=0.88)

plt.show()