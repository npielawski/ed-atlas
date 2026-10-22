from pymc.distributions.dist_math import check_parameters
from pytensor import tensor as pt
import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import patsy
import pymc as pm
import time
from quantiles import weighted_quantile

base_model = False
skip_figures = False

print("JAX configuration")
print("default backend   :", jax.default_backend())
print("devices           :", jax.devices())
print("local device count:", jax.local_device_count())
print()

print("Loading dataset...")
dataset_path = "datasets/mortality_dummies.parquet"
df = pl.read_parquet(dataset_path)
X = df.select(pl.all().exclude("count", "mean_mort30d"))
y = df["mean_mort30d"].to_numpy()
weights = df["count"].to_numpy()
print(f"Loaded {X.shape} dataset!")
print()

print("Computing age quantiles...")
q = [0.05, 0.35, 0.65, 0.95]
bins = np.round(weighted_quantile(X["ed_age"].to_numpy(), q, sample_weight=weights))
print("Age quantiles:", dict(zip(q, bins)))
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
        draws=8000,
        tune=1000,
        chains=1,
        nuts_sampler="numpyro"
    )

print("Saving trace...")
if base_model:
    idata.to_netcdf("models/trace_base.nc")
else:
    idata.to_netcdf("models/trace.nc")
(pl
    .DataFrame(X.columns)
    .write_csv("models/columns/columns_mortality.csv", include_header=False)
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

print("Testing model...")
betas = idata.posterior.betas.mean(("chain", "draw")).to_numpy()
p = pm.math.invlogit(pm.math.dot(phi, betas))
ll = weights * pm.logp(pm.Bernoulli.dist(p=p), y)
print(f"MAP: {ll.mean().eval():.3f}")

