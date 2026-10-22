import arviz as az
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import patsy
import scipy.stats as st
import xarray as xr
from snomed import load_snomed_en
from quantiles import weighted_quantile

language = "en"
output = "figures/risks"
fmt = "pdf"

print("Loading dataset...")
dataset_path = "datasets/mortality_dummies.parquet"
df = pl.read_parquet(dataset_path)
df = df.rename({col: col.replace("[", "").replace("]", "") for col in df.columns})
X = df.select(pl.all().exclude("count", "mean_mort30d"))
y = df["mean_mort30d"].to_numpy()
weights = df["count"].to_numpy()
print(f"Loaded {X.shape} dataset!")
print()

print("Computing age quantiles...")
q = [0.05, 0.35, 0.65, 0.95]
bins = weighted_quantile(X["ed_age"].to_numpy(), q, sample_weight=weights)
print("Age quantiles:", dict(zip(q, bins)))
print()

print("Loading snomed terms...")
snomed = load_snomed_en()

print("Loading models...")
idata = az.from_netcdf("models/trace.nc")
idata_base = az.from_netcdf("models/trace_base.nc")

all_symptoms = (pl
    .DataFrame({"dim_name": [col for col in X.columns if col.startswith("symptoms_")]})
    .with_columns(
        pl.col("dim_name").str.strip_prefix("symptoms_").alias("code")
    )
    .with_columns(
        pl.col("code").replace(old=snomed["code"], new=snomed["term"]).alias("term")
    )
)
n_symptoms = len(all_symptoms)
all_results, all_results_base = [], []
symptom_i = 0
while True:
    symptom = all_symptoms[symptom_i].to_dicts()[0]
    symptom_code, symptom_name, symptom_dim = symptom["code"], symptom["term"], symptom["dim_name"]
    print(f"Working out risk of death for symptom {symptom_name}")

    print("Preparing fake patient...")
    age = np.arange(18, 111)
    Xp = np.zeros((2 * len(age), X.shape[1]))
    Xp[:len(age), 0] = 1. # Sex
    Xp[:, 1] = np.tile(age, 2) # Age
    Xp[:, np.array(X.columns) == symptom["dim_name"]] = 1.

    dict_vars = dict(
        sex=Xp[:, 0],
        age=Xp[:, 1],
        k=bins[1:-1],
        lb=bins[0],
        ub=bins[-1],
        symptoms=Xp[:, 2:].astype(float)
    )
    D = patsy.dmatrix("sex * cr(age, knots=k, lower_bound=lb, upper_bound=ub) * symptoms", dict_vars)
    D = xr.DataArray(D, dims=["n", "dims"])
    D_base = patsy.dmatrix("sex * cr(age, knots=k, lower_bound=lb, upper_bound=ub)", dict_vars)
    D_base = xr.DataArray(D_base, dims=["n", "dims"])

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def relative_risk_logits(logp, logp_base):
        return (1 + np.exp(-logp_base)) / (1 + np.exp(-logp))

    logp = idata.posterior.betas @ D.T
    logp_base = idata_base.posterior.betas @ D_base.T

    cond_is_male = Xp[:, 0] > 0.5
    p_death = sigmoid(logp)
    p_death_mean = p_death.mean(("chain", "draw"))
    p_death_male = p_death_mean[cond_is_male]
    p_death_female = p_death_mean[~cond_is_male]
    p_death_ci = p_death.quantile([0.025, 0.975], ("chain", "draw"))
    p_death_male_lo, p_death_male_hi = p_death_ci[:, cond_is_male]
    p_death_female_lo, p_death_female_hi = p_death_ci[:, ~cond_is_male]

    p_death_base = sigmoid(logp_base)
    p_death_base_mean = p_death_base.mean(("chain", "draw"))
    p_death_base_male = p_death_base_mean[cond_is_male]
    p_death_base_female = p_death_base_mean[~cond_is_male]

    rel_risk = relative_risk_logits(logp, logp_base)
    rel_risk_mean = rel_risk.mean(axis=(0, 1))
    rel_risk_male = rel_risk_mean[cond_is_male]
    rel_risk_female = rel_risk_mean[~cond_is_male]
    rel_risk_ci = np.quantile(rel_risk, [0.025, 0.975], axis=(0, 1))
    rel_risk_male_lo, rel_risk_male_hi = rel_risk_ci[:, cond_is_male]
    rel_risk_female_lo, rel_risk_female_hi = rel_risk_ci[:, ~cond_is_male]

    matplotlib.rc("font", size=12)
    fig, ax = plt.subplots(2, 1, figsize=(8.0, 11.0), sharex=True)
    if symptom_code == "nosct:other":
        fig.suptitle("Risk of 30 day mortality across age and sex\nNo main symptom recorded")
    else:
        fig.suptitle(f"Risk of 30 day mortality across age and sex\nMain symptom: {symptom_name}")

    ax[0].plot(age, p_death_male, color="tab:blue", label="Male", gid="plt-male-main")
    ax[0].plot(age, p_death_base_male, ls="--", alpha=0.6, color="tab:blue", label="Male (base rate)", gid="plt-male-base")
    ax[0].fill_between(age, p_death_male_lo, p_death_male_hi, color="tab:blue", alpha=0.2, label="Male 95% CI", gid="plt-male-ci")
    ax[0].plot(age, p_death_female, color="tab:pink", label="Female", gid="plt-female-main")
    ax[0].plot(age, p_death_base_female, ls="--", alpha=0.6, color="tab:pink", label="Female (base rate)", gid="plt-female-base")
    ax[0].fill_between(age, p_death_female_lo, p_death_female_hi, color="tab:pink", alpha=0.2, label="Female 95% CI", gid="plt-female-ci")

    ax[0].grid()
    ax[0].legend(loc="upper left", ncols=2)
    ax[0].set_xlim(18, 110)
    ax[0].set_ylim(-0.025)
    ax[0].set_xticks(np.arange(0, 110, 10))
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[0].set_ylabel("Risk of death\nwithin 30 days [%]")

    ax[1].plot(age, rel_risk_male, color="tab:blue", label="Male risk", gid="plt-male-risk")
    age0 = np.ones_like(age)
    ax[1].fill_between(
        age,
        rel_risk_male_lo,
        rel_risk_male_hi,
        color="tab:blue",
        alpha=0.2,
        label="Male 95% CI",
        gid="plt-male-risk-ci"
    )
    ax[1].plot(age, rel_risk_female, color="tab:pink", label="Female risk", gid="plt-female-risk")
    ax[1].fill_between(
        age,
        rel_risk_female_lo,
        rel_risk_female_hi,
        color="tab:pink",
        alpha=0.2,
        label="Female 95% CI",
        gid="plt-female-risk-ci"
    )
    ax[1].plot([18, 110], [1, 1], "k--")
    ax[1].set_xlim(18, 110)
    ax[1].grid()
    ax[1].legend(loc="upper left", ncols=2)
    ax[1].set_xlabel("Age [years]")
    ax[1].set_ylabel("Relative risk of death\nwithin 30 days")

    cond = X[symptom_dim] == 1
    if cond.sum() > 1:
        weights_norm = weights / weights.sum()
        density = st.gaussian_kde(X.filter(cond)["ed_age"], weights=weights_norm[cond])
        ylim = ax[1].get_ylim()
        histHeight = ylim[1] * 0.03
        im = ax[1].imshow(
            np.expand_dims(density(age), axis=0),
            aspect="auto",
            origin="upper",
            interpolation="none",
            extent=(18, 110, -histHeight, 0.),
            vmin=0,
            vmax=density(age).max(),
            cmap="Oranges"
        )
        ax[1].set_ylim(-histHeight, ylim[1])

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15)

        cbar_ax = fig.add_axes([0.15, 0.07, 0.8, 0.015]) # [left, bottom, width, height]
        cbar = plt.colorbar(
            im,
            cax=cbar_ax,
            orientation="horizontal",
            aspect=30,
            format="{x:.2%}"
        )
        N_occur = int(cond.cast(pl.Float32) @ weights)
        cbar.set_label(f"Density of ED visits (N={N_occur:,})")
    else:
        plt.tight_layout()

    symptom_name = symptom_name.replace("/", "_")
    if language == "en":
        plt.savefig(f"{output}_en/risk_{symptom_code}.{fmt}")
    else:
        plt.savefig(f"{output}_en/risk_{symptom_code}.{fmt}")


    indices = np.arange(p_death.shape[1])
    np.random.shuffle(indices)
    for i in range(50):
        idx = indices[i]
        ax[0].plot(age, p_death[0, idx, cond_is_male], color="tab:blue", alpha=0.3, gid=f"plt-sample-main-male-{i}")
        ax[0].plot(age, p_death[0, idx, ~cond_is_male], color="tab:pink", alpha=0.3, gid=f"plt-sample-main-female-{i}")
        ax[1].plot(age, rel_risk[0, idx, cond_is_male], color="tab:blue", alpha=0.3, gid=f"plt-sample-risk-male-{i}")
        ax[1].plot(age, rel_risk[0, idx, ~cond_is_male], color="tab:pink", alpha=0.3, gid=f"plt-sample-risk-female-{i}")
    if language == "en":
        plt.savefig(f"{output}_svg_en/risk_{symptom_code}.svg")
    else:
        plt.savefig(f"{output}_svg/risk_{symptom_code}.svg")

    plt.close()

    all_results.append(pl.DataFrame({
        "symptom": symptom_code,
        "term": symptom_name,
        "is_male": Xp[:, 0].astype(bool),
        "ed_age": Xp[:, 1].astype(int),
        "mu": logp.mean("draw")[0].to_numpy(),
        "sigma": logp.std("draw")[0].to_numpy()
    }))
    all_results_base.append(pl.DataFrame({
        "symptom": symptom_code,
        "term": symptom_name,
        "is_male": Xp[:, 0].astype(bool),
        "ed_age": Xp[:, 1].astype(int),
        "mu": logp_base.mean("draw")[0].to_numpy(),
        "sigma": logp_base.std("draw")[0].to_numpy()
    }))

    symptom_i += 1
    if symptom_i == n_symptoms:
       break

all_results = pl.concat(all_results)
all_results.write_csv("models/static_mort_predictor.csv")

all_results_base = pl.concat(all_results_base)
all_results_base.write_csv("models/static_base_mort_predictor.csv")
