import sys
from test_toolkit import compute_metrics, plot_results
import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import patsy
import pickle
import pymc as pm
import time

N_models = "all"
if len(sys.argv[1]) == 2:
    is_chapter = True
    chapter = int(sys.argv[1])
    diagnosis = f"{chapter:02d}"
else:
    is_chapter = False
    diagnosis = sys.argv[1]

cmap = {
    "01": "#007145", "02": "#3d9a69", "03": "#9aeba6", "04": "#75be75", "05": "#8e690c", 
    "06": "#ebba59", "07": "#d77961", "08": "#aa5951", "09": "#ffa6a6", "10": "#8a455d", 
    "11": "#db79a6", "12": "#ffb6e3", "13": "#96558e", "14": "#aa7dca", "15": "#595da6", 
    "16": "#8aceff", "17": "#2d6971", "18": "#6db2ba", "19": "#18928a", "20": "#59d2b2", 
}

print("JAX configuration")
print("default backend   :", jax.default_backend())
print("devices           :", jax.devices())
print("local device count:", jax.local_device_count())
print()

print("Loading dataset...")
print(f"Working with diagnosis \"{diagnosis}\".")
df = (pl
    .read_parquet("datasets/diagnoses_dummies.parquet")
    .select(pl.col("ed_age", "is_male", "symptoms", "count", diagnosis))
    .group_by("ed_age", "is_male", "symptoms", diagnosis)
    .agg(count=pl.col("count").sum())
)
X = df.select(pl.col("ed_age", "is_male", "symptoms")).to_dummies("symptoms")
y = df[diagnosis].cast(pl.Float32).to_numpy()
weights = df["count"].to_numpy()
print(f"Loaded {X.shape} dataset!")
print()

print("Computing age quantiles...")
q = [0.05, 0.35, 0.65, 0.95]
bins = np.array([22.0, 43.0, 65.0, 88.0])
print("Age quantiles:", dict(zip(q, bins)))
print("UPDATE IF CHANGE IN MORTALITY EXPERIMENT.")
print()

print("Specifying interactions...")
start = time.time()
formula = "sex * cr(age, knots=k, lower_bound=lb, upper_bound=ub) * symptoms"
print(f"Formula: y ~ {formula}")
D = patsy.dmatrix(
    formula,
    {
        "sex": X["is_male"].to_numpy(),
        "age": X["ed_age"].to_numpy(),
        "k": bins[1:-1],
        "lb": bins[0],
        "ub": bins[-1],
        "symptoms": X.select(pl.col("^symptoms_.*$")).cast(pl.Float32).to_numpy()
    }
)
end = time.time()
design_info = D.design_info
print(f"Time: {end - start:.3f} seconds")
print(f"Dimensions: {D.shape[1]}")
D = np.asarray(D, dtype="float32")
print()

print("Loading models...")
idata = az.from_netcdf(f"models/trace_diag_{diagnosis}.nc")
print()

print("Testing model...")
dims = idata.posterior.betas.shape[-1]
if N_models == "all":
    betas = idata.posterior.betas.to_numpy().reshape(-1, dims)
else:
    betas = idata.posterior.betas.to_numpy().reshape(-1, dims)[:N_models, :]
p = pm.math.invlogit(pm.math.dot(D, betas.T))

betas_mean = idata.posterior.betas.mean(("chain", "draw")).to_numpy()
p_mean = pm.math.invlogit(pm.math.dot(D, betas_mean))

y_pred = p.eval()
y_pred_cols = [f"y_pred_{i:04d}" for i in range(y_pred.shape[1])]
results = (pl
    .DataFrame({
        "y_mean_pred": p_mean.eval(),
        "y_pred": y_pred.tolist(),
        "dead": (weights*y).astype("uint32"),
        "alive": (weights*(1-y)).astype("uint32")
    })
    .with_columns(
        pl.col("y_pred").list.to_struct(fields=y_pred_cols)
    )
    .unnest("y_pred")
)
results = pl.concat([
    results
        .filter(pl.col("dead") != 0)
        .select(
            "y_mean_pred",
            "^y_pred_.*$",
            pl.lit(1).alias("y_true"),
            pl.col("dead").alias("weight")
        ),
    results
        .filter(pl.col("alive") != 0)
        .select(
            "y_mean_pred",
            "^y_pred_.*$",
            pl.lit(0).alias("y_true"),
            pl.col("alive").alias("weight")
        )
])

scores = compute_metrics(results)

for score_name, (mean, ci_lo, ci_hi) in scores["metrics"].items():
    if score_name in ["ROC", "AP", "AP_chance_level"]:
        print(f"SwED {score_name} score: {100*mean:.2f} (95% CI: [{100*ci_lo:.2f}, {100*ci_hi:.2f}])")
        continue
    print(f"SwED {score_name} score: {mean:.3f} (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")

if is_chapter:
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax2 = ax[1, 0].twinx()

    plot_results((ax, ax2), scores, results, dataset_name="SwED", color=cmap[diagnosis], partial=True)

    plt.tight_layout()
    plt.savefig(f"figures/test/test_diag_{diagnosis}.png")

    with open(f"figures/test/test_diag_{diagnosis}.pickle", "wb") as handle:
        pickle.dump(
            dict(fig=fig, ax=ax, ax2=ax2),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL
        )

