import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rc("font", size=15)

# Loading map of Sweden
sweden = gpd.read_file("data/maps/swedish_municipalities.geojson")

# Loading true population counts
population = pd.read_csv(
    "data/maps/population_counts_2015.csv",
    skiprows=2,
    dtype={"kommun": str},
    encoding="ISO-8859-1"
)
# split "region" by space, with the first part containing the region id
population["id"] = population["region"].str.split(" ", n=1, expand=True)[0]
population["count"] = population["2015"]
population["count"] += 1
population["prob"] = population["count"] / population["count"].sum()

# Loading patient counts
patients = pd.read_csv("data/maps/kommun_counts.csv", dtype={"kommun": str})
patients = patients.rename(columns={"kommun": "id"})
patients["count"] += 1
patients["prob"] = patients["count"] / patients["count"].sum()

# Full merge
merged = sweden.merge(population, left_on="id", right_on="id", how="left")
merged = merged.merge(patients, on="id", suffixes=("_pop", "_pat"))
merged["logcounts_pat"] = np.log10(merged["count_pat"])
merged["logratio"] = np.log10(merged["prob_pat"] / merged["prob_pop"])
vmin, vmax = merged["logratio"].min(), merged["logratio"].max()

fig, ax = plt.subplots(1, 2, figsize=(7*2, 10))

# (a) Patients per municipality
plot = merged.plot(
    column="logcounts_pat",
    cmap="viridis",
    ax=ax[0],
)
ax[0].set_axis_off()

cbar = plt.colorbar(plot.collections[0], ax=ax[0])
cbar.set_label("Count of patients per municipality")
log_ticks = np.arange(3, merged["logcounts_pat"].max()).tolist()
log_ticks.insert(0, merged["logcounts_pat"].min())
log_ticks.append(merged["logcounts_pat"].max())
log_ticks = np.array(log_ticks)
cbar.set_ticks(log_ticks)
cbar.set_ticklabels([f"{int(10**x):,}" for x in log_ticks])

# (b) Patients relative to the municipality population
norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
plot = merged.plot(
    column="logratio",
    cmap="berlin",
    ax=ax[1],
    norm=norm,
)
ax[1].set_axis_off()

cbar = plt.colorbar(plot.collections[0], ax=ax[1])
cbar.set_label("Number of patients to 100,000 people in the municipality")
log_ticks = norm.inverse(np.linspace(0, 1, num=7))
cbar.set_ticks(log_ticks)
cbar.set_ticklabels([f"{int(10**x*100_000):,}" for x in log_ticks])

plt.text(0.84, 0.5,
    "Fewer patients than population ↔︎ More patients than population",
    transform=fig.transFigure,
    rotation=90, va="center", ha="left",
    color="#555"
)

plt.tight_layout()
plt.savefig("figures/patient_counts.svg")
