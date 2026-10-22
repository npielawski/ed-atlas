from collections import OrderedDict
import glob
import heapq
import sys
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import patsy
import polars as pl
import scipy.stats as st
import xarray as xr
from tqdm import tqdm
from colorscheme import cmap_glasbey
from icd import load_icd
from snomed import load_snomed_en

output = "figures/diags"
fmt = "pdf"
short_legend = False

# symptom_to_plot = 21522001 # Abdominal pain
# symptom_to_plot = 106004004 # Hemorrhagic complication of pregnancy
symptom_to_plot = "args"

# language = "se"
language = "en"

plot_level = sys.argv[1]
if plot_level not in ["chapters", "blocks", "codes"]:
    print(f"plot_level was set to {plot_level} and that's just wrong.")
    sys.exit()

topk = 20

print("Loading dataset...")
df = pl.read_parquet("datasets/diagnoses.parquet").select(pl.all().exclude("mean_mort30d"))
X = df.select(pl.col("ed_age", "is_male", "symptoms")).to_dummies("symptoms")
weights = df["count"].to_numpy()
print(f"Loaded {X.shape} dataset!")
print()

print("Computing age quantiles...")
q = [0.05, 0.35, 0.65, 0.95]
bins = np.array([22.0, 43.0, 65.0, 88.0])
print("Age quantiles:", dict(zip(q, bins)))
print("UPDATE IF CHANGE IN MORTALITY EXPERIMENT.")
print()

print("Loading snomed terms...")
snomed = load_snomed_en()

print("Loading models...")
model_paths = glob.glob("models/trace_diag_[!base]*")
idata = OrderedDict()
columns_subset = OrderedDict()
for model_path in tqdm(sorted(model_paths)):
    diag_name = model_path.split("_")[-1][:-3]
    if plot_level == "chapters" and len(diag_name) != 2:
        continue
    if plot_level == "blocks":
        if len(diag_name) != 7:
            continue
        if diag_name in [
                "C00-C75", "C00-C97", "D37-D48", "M00-M25", "M40-M54",
                "M60-M79", "M80-M94", "T20-T32", "V01-V99", "V01-X59",
                "W00-X59", "Y40-Y84"
            ]:
            continue
    if plot_level == "codes" and len(diag_name) != 3:
        continue
    idata[diag_name] = az.from_netcdf(model_path)
    try:
        columns_subset[diag_name] = pl.read_csv(
            f"models/columns/columns_{diag_name}.csv",
            has_header=False
        )
    except FileNotFoundError:
        print(f"Did not find {diag_name} csv file, check time error in logs!")
        print("Removing model.")
        del idata[diag_name]

print(f"Loaded {len(idata)} models!")

print("Loading symptoms and diagnosis mappings...")
all_symptoms = (pl
    .DataFrame(
        {"code": df["symptoms"].unique().to_list()}
    )
    .with_columns(
        ("symptoms_" + pl.col("code")).alias("dim_name")
    )
    .with_columns(
        pl.col("code").replace(old=snomed["code"], new=snomed["term"]).alias("term")
    )
    .sort("code")
)
all_diagnoses = load_icd()

n_symptoms = len(all_symptoms)
if symptom_to_plot != "args":
    symptom_i = np.argwhere(all_symptoms["code"] == symptom_to_plot).item()
else:
    symptom_i = int(sys.argv[2])

while True:
    symptom = all_symptoms[symptom_i].to_dicts()[0]
    symptom_code, symptom_name, symptom_dim = symptom["code"], symptom["term"], symptom["dim_name"]
    if symptom_name == "nosct:other":
        print("Working out risk of death for no symptom")
    else:
        print(f"Working out risk of death for symptom {symptom_name}")

    logp_male, logp_female = OrderedDict(), OrderedDict()
    p_diagnosis_male, p_diagnosis_female = OrderedDict(), OrderedDict()
    p_diagnosis_male_lo, p_diagnosis_male_hi = OrderedDict(), OrderedDict()
    p_diagnosis_female_lo, p_diagnosis_female_hi = OrderedDict(), OrderedDict()
    for diagnosis in idata.keys():
        print(f"Preparing fake patient for {diagnosis}...")
        columns = columns_subset[diagnosis]
        age = np.arange(18, 111)
        Xp = np.zeros((2 * len(age), len(columns)))
        Xp[:len(age), 0] = 1. # Sex
        Xp[:, 1] = np.tile(age, 2) # Age
        Xp[:, columns.to_numpy()[:, 0] == symptom["dim_name"]] = 1.

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

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        logp = idata[diagnosis].posterior.betas @ D.T

        cond_is_male = Xp[:, 0] > 0.5
        p_diagnosis = sigmoid(logp)
        p_diagnosis_mean = p_diagnosis.mean(("chain", "draw"))
        p_diagnosis_male[diagnosis] = p_diagnosis_mean[cond_is_male]
        p_diagnosis_female[diagnosis] = p_diagnosis_mean[~cond_is_male]
        logp_male[diagnosis] = logp[..., cond_is_male]
        logp_female[diagnosis] = logp[..., ~cond_is_male]
        p_diagnosis_ci = p_diagnosis.quantile([0.025, 0.975], ("chain", "draw"))
        p_diagnosis_male_lo[diagnosis], p_diagnosis_male_hi[diagnosis] = p_diagnosis_ci[:, cond_is_male]
        p_diagnosis_female_lo[diagnosis], p_diagnosis_female_hi[diagnosis] = p_diagnosis_ci[:, ~cond_is_male]

    if topk is not None:
        female_max_proba = OrderedDict(zip(
            p_diagnosis_female.keys(),
            map(
                lambda x: sigmoid(x[..., age<100]).max("n").median(("chain", "draw")).item(),
                logp_female.values()
            )
        ))
        male_max_proba = OrderedDict(zip(
            p_diagnosis_male.keys(),
            map(
                lambda x: sigmoid(x[..., age<100]).max("n").median(("chain", "draw")).item(),
                logp_male.values()
            )
        ))
        def topk_from_two_dicts(dict1, dict2, k):
            # Merge dictionaries, keeping the highest value for overlapping keys
            combined = dict1.copy()
            for key, value in dict2.items():
                if key in combined:
                    combined[key] = max(combined[key], value)
                else:
                    combined[key] = value

            # Retrieve the top k items with the highest values
            top_k = heapq.nlargest(k, combined.items(), key=lambda x: x[1])

            # Return only the keys
            return [key for key, _ in top_k]
        labels_to_use = topk_from_two_dicts(female_max_proba, male_max_proba, topk)
    else:
        labels_to_use = list(idata.keys())
    print("Chosen labels:", labels_to_use)

    matplotlib.rc("font", size=12)
    fig, (axF, axM) = plt.subplots(2, 1, figsize=(15.0, 4.8*3), sharex=True)
    if symptom_name == "nosct:other":
        fig.suptitle("Probability of diagnoses across age and sex\nNo main symptom recorded")
    else:
        fig.suptitle(f"Probability of diagnoses across age and sex\nMain symptom: {symptom_name}")

    ci_alpha = 0.4
    grid_alpha = 0.3
    cmap = cmap_glasbey
    if plot_level != "chapters":
        cmap = list(cmap.values())

    legend_margin = 0.02
    legend_size = 0.40
    plot_to_legend = 0.50
    colorbar_shift = -0.04

    axF.grid(alpha=grid_alpha)
    if plot_level == "chapters":
        labels = (pl
            .DataFrame({"code": labels_to_use})
            .join(all_diagnoses, how="left", on="code")
            .select(
                "term_en",
                pl.col("term").str.extract_groups(r"(?<term>.*) \((?<range>.*)\)")
            )
            .select(
                ("$\\tt{" + pl.col("term").struct["range"] + "}$: " + pl.col("term").struct["term"]).alias("label")
                if language == "se" else
                ("$\\tt{" + pl.col("term").struct["range"] + "}$: " + pl.col("term_en")).alias("label")
            )
        )["label"]
        if short_legend and symptom_to_plot == 21522001: # Only for abdominal pain
            # Just for generating a legend that is not too wide for the journal
            labels[8] = labels[8].replace("and conn", "and\nconn")
            labels[9] = labels[9].replace("ences of", "ences\nof")
            labels[-5] = labels[-5].replace("and cert", "and\ncert")
            labels[-2] = labels[-2].replace("and chrom", "and\nchrom")
    else:
        labels = (pl
            .DataFrame({"code": labels_to_use})
            .join(all_diagnoses, how="left", on="code")
            .with_columns(
                ("$\\tt{" + pl.col("code") + "}$: " + pl.col("term_en")).alias("label")
            )
        )["label"]
        if short_legend and symptom_to_plot == 21522001: # Only for abdominal pain
            if plot_level == "blocks":
                labels[5] = labels[5].replace("to preg", "to\npreg")
                labels[-8] = labels[-8].replace("psycho", "psycho-\n")
                labels[-3] = labels[-3].replace("and uns", "and\nuns")
            else: # plot_level == "codes"
                labels[7] = labels[7].replace("and unsp", "and\nunsp")
                labels[8] = labels[8].replace("and broad", "and\nbroad")
                labels[-4] = labels[-4].replace("genital o", "genital\no")

    baseline = np.zeros_like(list(p_diagnosis_female.values())[0])
    for i, k in enumerate(labels_to_use):
        axF.fill_between(
            age,
            baseline,
            baseline + p_diagnosis_female[k],
            label=labels[i],
            color=cmap[k if plot_level == "chapters" else i],
            gid=f"female-{i}"
        )
        baseline += p_diagnosis_female[k]
    maxF = baseline.max()
    axF.set_xticks(np.arange(0, 120, 10))
    axF.set_xlim(18, 110)
    axF.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axF.set_ylabel("Female probability of diagnosis\nwithin 30 days [%]")

    axM.invert_yaxis()
    axM.grid(alpha=grid_alpha)

    baseline = np.zeros_like(list(p_diagnosis_male.values())[0])
    for i, k in enumerate(labels_to_use):
        axM.fill_between(
            age,
            baseline,
            baseline + p_diagnosis_male[k],
            color=cmap[k if plot_level == "chapters" else i],
            gid=f"male-{i}"
        )
        baseline += p_diagnosis_male[k]
    maxM = baseline.max()
    axM.set_xticks(np.arange(0, 120, 10))
    axM.set_xlim(18, 110)
    axM.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axM.set_ylabel("Male probability of diagnosis\nwithin 30 days [%]")
    axM.set_xlabel("Age [years]")

    # https://stackoverflow.com/a/66154843
    maxHeight = max(maxF, maxM) * (plt.margins()[1] + 1)

    axF.text(20, maxHeight * 0.95, "♀", size=40, alpha=.4, va="top")
    axM.text(20, maxHeight * 0.98, "♂", size=40, alpha=.4, va="bottom")

    cond = X[symptom_dim] == 1
    cond_female = cond & (~X["is_male"])
    cond_male = cond & X["is_male"]
    weights_norm = weights / weights.sum()

    is_female_kde_ok = cond_female.sum() > 1
    is_male_kde_ok = cond_male.sum() > 1
    if is_female_kde_ok:
        try:
            densityF = st.gaussian_kde(X.filter(cond_female)["ed_age"], weights=weights_norm[cond_female])
            histF = densityF(age)
        except np.linalg.LinAlgError:
            print("OOps! LinAlgError with female kde!")
            is_female_kde_ok = False
    if is_male_kde_ok:
        try:
            densityM = st.gaussian_kde(X.filter(cond_male)["ed_age"], weights=weights_norm[cond_male])
            histM = densityM(age)
        except np.linalg.LinAlgError:
            print("OOps! LinAlgError with male kde!")
            is_male_kde_ok = False

    if is_female_kde_ok:
        im = axF.imshow(
            np.expand_dims(histF, axis=0),
            aspect="auto",
            origin="upper",
            interpolation="none",
            extent=(18, 110, maxHeight*0.97, maxHeight),
            vmin=0,
            vmax=max(histF.max(), histM.max()) if is_male_kde_ok else histF.max(),
            cmap="Oranges"
        )
    if is_male_kde_ok:
        im = axM.imshow(
            np.expand_dims(histM, axis=0),
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(18, 110, maxHeight*0.97, maxHeight),
            vmin=0,
            vmax=max(histF.max(), histM.max()) if is_female_kde_ok else histM.max(),
            cmap="Oranges"
        )

    axF.set_ylim(0., maxHeight)
    axM.set_ylim(maxHeight, 0.)

    yticklabels = axM.get_yticklabels()
    yticklabels[0] = "" # Avoid double 0.0% label on y ticks
    axM.set_yticklabels(yticklabels)

    fig.legend(title=plot_level.capitalize(), loc="upper center", bbox_to_anchor=(0.5, legend_size))

    if is_female_kde_ok or is_male_kde_ok:
        cbar_ax = fig.add_axes([0.15, legend_size-colorbar_shift, 0.7, 0.015]) # [left, bottom, width, height]
        cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal", aspect=30, format="{x:.2%}")
        N_occur = int(cond.cast(pl.Float32) @ weights)
        cbar.set_label(f"Density of diagnoses (N={N_occur:,})")

    plt.tight_layout()
    fig.subplots_adjust(bottom=plot_to_legend, hspace=0)
    symptom_name = symptom_name.replace("/", "_")
    if language == "se":
        plt.savefig(f"{output}_{plot_level}/diag_{symptom_code}.{fmt}")
        print(f"saved image: {output}_{plot_level}/diag_{symptom_code}.{fmt}")
    elif short_legend:
        plt.savefig(f"{output}_{plot_level}_en/diag_{symptom_code}_short.{fmt}")
        print(f"saved image: {output}_{plot_level}_en/diag_{symptom_code}_short.{fmt}")
    else:
        plt.savefig(f"{output}_{plot_level}_en/diag_{symptom_code}.{fmt}")
        print(f"saved image: {output}_{plot_level}_en/diag_{symptom_code}.{fmt}")

    baseline = np.zeros_like(list(p_diagnosis_female.values())[0])
    for i, k in enumerate(labels_to_use):
        c = cmap[k if plot_level == "chapters" else i]
        axF.plot(age, baseline + p_diagnosis_female_lo[k], color=c, lw=2, ls="--", zorder=3, gid=f"female-{i}-lo")
        axF.plot(age, baseline + p_diagnosis_female_hi[k], color=c, lw=2, ls="--", zorder=3, gid=f"female-{i}-hi")
        baseline += p_diagnosis_female[k]
    baseline = np.zeros_like(list(p_diagnosis_male.values())[0])
    for i, k in enumerate(labels_to_use):
        c = cmap[k if plot_level == "chapters" else i]
        axM.plot(age, baseline + p_diagnosis_male_lo[k], color=c, lw=2, ls="--", zorder=3, gid=f"male-{i}-lo")
        axM.plot(age, baseline + p_diagnosis_male_hi[k], color=c, lw=2, ls="--", zorder=3, gid=f"male-{i}-hi")
        baseline += p_diagnosis_male[k]

    if language == "se":
        plt.savefig(f"{output}_{plot_level}_svg/diag_{symptom_code}.svg")
        print(f"saved image: {output}_{plot_level}_svg/diag_{symptom_code}.svg")
    elif short_legend:
        plt.savefig(f"{output}_{plot_level}_svg_en/diag_{symptom_code}_short.svg")
        print(f"saved image: {output}_{plot_level}_svg_en/diag_{symptom_code}_short.svg")
    else:
        plt.savefig(f"{output}_{plot_level}_svg_en/diag_{symptom_code}.svg")
        print(f"saved image: {output}_{plot_level}_svg_en/diag_{symptom_code}.svg")

    plt.close()

    import json
    for ageIdx, actualAge in enumerate(age):
        probas = dict(
            p_diag={k: v[ageIdx].item() for k, v in p_diagnosis_female.items()},
            p_lo={k: v[ageIdx].item() for k, v in p_diagnosis_female_lo.items()},
            p_hi={k: v[ageIdx].item() for k, v in p_diagnosis_female_hi.items()},
        )
        with open(f"models/diagnoses/{plot_level}_{symptom_code}_{actualAge}_F.json", "w+") as f:
            f.write(json.dumps(probas, separators=(',', ':')))
        probas = dict(
            p_diag={k: v[ageIdx].item() for k, v in p_diagnosis_male.items()},
            p_lo={k: v[ageIdx].item() for k, v in p_diagnosis_male_lo.items()},
            p_hi={k: v[ageIdx].item() for k, v in p_diagnosis_male_hi.items()},
        )
        with open(f"models/diagnoses/{plot_level}_{symptom_code}_{actualAge}_M.json", "w+") as f:
            f.write(json.dumps(probas, separators=(',', ':')))

    symptom_i += 1
    if symptom_i == n_symptoms:
       break
    break

