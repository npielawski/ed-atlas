import os
import sys
from pymc.distributions.dist_math import check_parameters
import polars as pl
import polars.selectors as cs
from pytensor import tensor as pt
import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import patsy
import pymc as pm
import time

base_model = False
skip_figures = True
overwrite = False

print("JAX configuration")
print("default backend   :", jax.default_backend())
print("devices           :", jax.devices())
print("local device count:", jax.local_device_count())
print()

print("Loading dataset...")
df = pl.read_parquet("datasets/diagnoses_dummies_pruned.parquet")
diagnosis = sorted([
    col
    for col in df.select(cs.boolean().exclude("is_male")).columns
    if len(col) != 4
])[int(sys.argv[1])]
print(f"Working with diagnosis \"{diagnosis}\".")
if not overwrite:
    if base_model:
       path_to_check = f"models/trace_diag_base_{diagnosis}.nc"
    else:
        path_to_check = f"models/trace_diag_{diagnosis}.nc"
    if os.path.exists(path_to_check):
        print("Diagnosis already covered! Cannot overwrite! Exiting!")
        sys.exit()
df = (pl
    .read_parquet("datasets/diagnoses_dummies.parquet")
    .select(pl.col("ed_age", "is_male", "symptoms", "count", diagnosis))
    .group_by("ed_age", "is_male", "symptoms", diagnosis)
    .agg(count=pl.col("count").sum())
)
X = df.select(pl.col("ed_age", "is_male", "symptoms")).to_dummies("symptoms")
y = df[diagnosis].cast(pl.Float32)
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
formula = "sex * cr(age, knots=k, lower_bound=lb, upper_bound=ub)"
if not base_model:
    formula += " * symptoms"
print(f"Formula: y ~ {formula}")
D = patsy.dmatrix(
    formula,
    {
        "sex": X["is_male"].to_numpy(),
        "age": X["ed_age"],
        "k": bins[1:-1],
        "lb": bins[0],
        "ub": bins[-1],
        "symptoms": X.select(pl.col("^symptoms_.*$")).cast(pl.Float32)
    }
)
end = time.time()
design_info = D.design_info
print(f"Time: {end - start:.3f} seconds")
print(f"Dimensions: {D.shape[1]}")
D = np.asarray(D, dtype="float32")
print()


print("Preparing model and sampling...")
n_dims = len(design_info.column_names)
COORDS = {
    "N": np.arange(D.shape[0]),
    "dims": design_info.column_names,
}
prior_mu = np.zeros(D.shape[1])
prior_mu[0] = -np.log(9)

def bernoulli_logp(value, p, weights):
    res = pt.switch(
        pt.or_(pt.lt(value, 0), pt.gt(value, 1)),
        -np.inf,
        pt.switch(
            pt.or_(pt.eq(value, 0.0), pt.eq(value, 1.0)),
            pt.switch(value, pt.log(p), pt.log1p(-p)),
            value * pt.log(p) + (1 - value) * pt.log1p(-p)
        )
    )

    return weights * check_parameters(
        res,
        0 <= p,
        p <= 1,
        msg="0 <= p <= 1",
    )

def bernoulli_rvs(p, weights, rng, size):
    return np.random.binomial(1, p, size)

with pm.Model(coords=COORDS) as model:
    phi = pm.Data("phi", D)
    betas = pm.Normal("betas", mu=prior_mu, sigma=0.5, dims=("dims",))
    logp = pm.Deterministic("logp", pm.math.dot(phi, betas))
    p = pm.math.invlogit(logp)
    pm.CustomDist(
        "obs",
        p,
        weights,
        random=bernoulli_rvs,
        logp=bernoulli_logp,
        observed=y,
    )

    idata = pm.sample(
        draws=2000,
        tune=500,
        chains=2,
        nuts_sampler="numpyro"
    )

print("Saving trace...")
if "logp" in idata.posterior:
    del idata.posterior["logp"]
if base_model:
    idata.to_netcdf(f"models/trace_diag_base_{diagnosis}.nc")
else:
    idata.to_netcdf(f"models/trace_diag_{diagnosis}.nc")
(pl
    .DataFrame(X.columns)
    .write_csv(f"models/columns/columns_{diagnosis}.csv", include_header=False)
)
print()

if base_model and not skip_figures:
    plot_vars = dict(
        var_names=["betas"],
        show=False
    )
    print("Generating figures...")
    print(".", end="")
    az.plot_ess(idata, kind="evolution", **plot_vars)
    plt.savefig("figures/plot_ess.png")
    print(".", end="")
    az.plot_forest(idata, **plot_vars)
    plt.savefig("figures/plot_forest.png")
    print(".", end="")
    az.plot_pair(idata, **plot_vars)
    plt.savefig("figures/plot_pair.png")
    print(".", end="")
    az.plot_trace(idata, compact=False, **plot_vars)
    plt.savefig("figures/plot_trace.png")
    print(".", end="")
    az.plot_trace(idata, compact=False, kind="rank_bars", **plot_vars)
    plt.savefig("figures/plot_trace_rank.png")
    print(".")
    az.summary(idata).to_csv("figures/summary.csv")
else:
    print("Skipping figures!")
print()

