import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend had to do this for support on different computers initially, YMMV

# Load the dataset
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Filter for the contiguous United States (excluding Alaska)
contiguous_us = world[(world.name == "United States of America") & (world.continent == "North America")]
contiguous_us = contiguous_us[~contiguous_us.geometry.simplify(0.1).centroid.map(lambda x: x.coords[0][0] < -130)]

# Filter for Alaska
alaska = world[world.name == "United States of America"]
alaska = alaska[alaska.geometry.simplify(0.1).centroid.map(lambda x: x.coords[0][0] < -130)]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(15, 8))

# Plot Contiguous United States
contiguous_us.plot(ax=axs[0], color='lightgrey', edgecolor='black')
axs[0].set_title("Contiguous United States")
axs[0].set_axis_off()

# Plot Alaska
alaska.plot(ax=axs[1], color='lightgrey', edgecolor='black')
axs[1].set_title("Alaska")
axs[1].set_axis_off()

plt.tight_layout()
plt.savefig("map1.png")
