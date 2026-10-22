"""Builds the data files of the ED Atlas web application (ed-atlas.org).

Reads the aggregated datasets (generate_dataset.py) and the static predictors
(plot_risk_pymc.py), and writes JSON files to web_data/:

    symptoms.json            deaths and population per presenting complaint
    diagnoses.json           diagnoses with differentially private survivor counts
    static_model.json        mortality model, per symptom, age and sex
    static_base_model.json   mortality model without symptoms
    index_symptoms.json      presenting complaints with their English terms
    index_diagnoses.json     ICD-10 codes with their English terms
"""
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
import json
import secrets

import arviz as az
import numpy as np
import polars as pl
import polars.selectors as cs
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import MultiLabelBinarizer

from icd import load_icd
from snomed import load_snomed_en

out = Path("web_data")
out.mkdir(exist_ok=True)


def logp_ab(value):
    # Jeffreys-like prior on the Beta hyperparameters
    return pt.log(pt.pow(pt.sum(value), -5/2))


# *** SYMPTOMS ***
df_mort = pl.read_parquet("datasets/mortality.parquet")
snomedct = load_snomed_en()
df_mort30 = (df_mort
    .with_columns(
        pl.col("symptoms")
        .str.strip_chars("[]")
        .alias("id")
    )
    .with_columns(
        pl.col("id")
          .replace_strict(
              old=snomedct["code"],
              new=snomedct["term"],
              default=None
          )
          .alias("term")
    )
    .group_by("id", "term")
    .agg(
        count=pl.col("count").sum(),
        mort30d=(pl.col("mean_mort30d") * pl.col("count")).sum().cast(pl.UInt32)
    )
)

# Beta prior of the mortality risk across symptoms (empirical Bayes)
with pm.Model() as model:
    ab = pm.HalfFlat("ab", shape=2)
    pm.Potential("p(a, b)", logp_ab(ab))
    alpha = pm.Deterministic("alpha", ab[0])
    beta = pm.Deterministic("beta", ab[1])
    theta = pm.Beta("theta", alpha=ab[0], beta=ab[1], shape=len(df_mort30))
    pm.Binomial(
        "y",
        p=theta,
        observed=df_mort30["mort30d"].to_numpy(),
        n=df_mort30["count"].to_numpy()
    )
    ns = pm.Deterministic("ns", alpha + beta)
    mu = pm.Deterministic("mu", alpha / ns)

    idata = pm.sample(random_seed=0)
print(az.summary(idata, ["mu", "ns"]))

symptoms = json.dumps(
    df_mort30.rename({"count": "n_pop", "mort30d": "deaths"}).to_dicts(),
    separators=(",", ":")
)
(out / "symptoms.json").write_text(symptoms)


# *** DIAGNOSES: differential privacy ***
# Discrete Laplace sampler (Canonne, Kamath and Steinke, 2020), using a
# cryptographically secure random number generator.
def bernoulli_quotient_rvs(number):
    K = secrets.randbelow(number.denominator) + 1
    return 1 if K <= number.numerator else 0


def bernoulli_exp_rvs(number):
    if 0. <= number <= 1.:
        K = 1
        while True:
            A = bernoulli_quotient_rvs(number * Fraction(1, K))
            if A == 0:
                break
            else:
                K += 1
        return K % 2
    else:
        for k in range(1, int(number)+1):
            B = bernoulli_exp_rvs(Fraction(1))
            if B == 0:
                return 0
        return bernoulli_exp_rvs(number - int(number))


def discrete_laplace_rvs(number):
    number = Fraction(number)
    assert number.numerator >= 1
    assert number.denominator >= 1

    while True:
        U = secrets.randbelow(number.numerator)
        D = bernoulli_exp_rvs(Fraction(U, number.numerator))
        if D != 0:
            break
    V = 0
    while True:
        B = bernoulli_exp_rvs(Fraction(1))
        if B == 0:
            break
        else:
            V += 1
    X = U + number.numerator * V
    Y = int(X / number.denominator)
    B = bernoulli_quotient_rvs(Fraction(1, 2))
    if B == 1 and Y == 0:
        return discrete_laplace_rvs(number)
    else:
        return (1 - 2 * B) * Y


df_diag = (pl
    .read_parquet("datasets/mortality_unique.parquet")
    .with_columns(
        (pl.col("n_pop") - pl.col("n_dead")).alias("n_alive")
    )
)

icd2parent = (pl
    .read_csv(
        "mappings/icd-10-se.tsv",
        columns=["Kod", "Överordnad kod"],
        new_columns=["code", "parent"],
        separator="\t",
        schema_overrides=pl.Schema({"Kod": pl.String, "Överordnad kod": pl.String}),
        null_values=""
    )
    .drop_nulls()
    .with_columns(
        pl.all().str.replace(r"\.", "")
    )
)

# Lookup of parents and children. Codes ending with "_extra" hold the
# patients diagnosed with the code itself rather than with one of its children.
parent_dict = dict(zip(icd2parent["code"], icd2parent["parent"]))
icd2child = icd2parent.group_by("parent").agg("code")
child_dict = dict(zip(icd2child["parent"], icd2child["code"]))


def find_all_parents(code, parent_dict=parent_dict):
    if code.endswith("_extra"):
        return [code] + find_all_parents(code.split("_")[0], parent_dict)
    parents = [code]
    while parents[-1] is not None:
        parents.append(find_parent(parents[-1], parent_dict))
    return parents[:-1]


def find_parent(code, parent_dict=parent_dict):
    if code.endswith("_extra"):
        return code.split("_")[0]
    parent = parent_dict[code] if code in parent_dict else None
    if parent is None and len(code) > 2:
        return find_parent(code[:-1], parent_dict)
    else:
        return parent


def find_children(code, child_dict=child_dict):
    return child_dict[code].to_list() if code in child_dict else []


def find_depth(code, parent_dict=parent_dict):
    if code.endswith("_extra"):
        return find_depth(code.split("_")[0], parent_dict) + 1
    i = -1
    while code is not None:
        code = find_parent(code, parent_dict)
        i += 1
    return i


def combine_estimates(x, var_x, y, var_y):
    var_z = (var_x * var_y) / (var_x + var_y)
    z = var_z * (x / var_x + y / var_y)
    return z, var_z


# Survivor counts at every level of the ICD-10 tree
df = (df_diag
    .with_columns(
        pl.col("diagnosis").map_elements(find_all_parents, return_dtype=pl.List(pl.String)).alias("diag_parents")
    )
    .with_columns(pl.col("diag_parents").list.len().alias("tree_depth") - 1)
    .sort("tree_depth", "diagnosis", descending=True)
)
result = OrderedDict()
for diagnosis, n_dead, n_pop, n_alive, _, tree_depth in df.iter_rows():
    parents = find_all_parents(diagnosis)
    result[diagnosis + "_extra"] = (n_dead, n_alive)

    for p in parents:
        if p in result:
            prev_n_dead, prev_n_alive = result[p]
            result[p] = (prev_n_dead+n_dead, prev_n_alive+n_alive)
        else:
            result[p] = (n_dead, n_alive)

# Adding noise to the survivor counts
epsilon, sensitivity = 2, max(len(find_all_parents(d)) for d in result.keys())
alpha = np.exp(-epsilon / sensitivity)
n_alive_variance = 2 * alpha / (1 - alpha)**2
print(f"{epsilon=}")
print(f"{sensitivity=}")
print(f"{n_alive_variance=}")

result_noisy = result.copy()
for diag, row in result_noisy.items():
    noisy_count = row[1] + discrete_laplace_rvs(Fraction(sensitivity, epsilon))
    result_noisy[diag] = (row[0], noisy_count, n_alive_variance)

# Combining the noisy estimates of parents and children
x = OrderedDict({k: v[1] for k, v in result_noisy.items()})
var = OrderedDict({k: n_alive_variance for k, v in result_noisy.items()})
z_bigup = x.copy()
var_bigup = var.copy()
n_dead = OrderedDict({k: v[0] for k, v in result_noisy.items()})
for diag in sorted(x.keys(), key=lambda k: find_depth(k)):
    children = [c for c in find_children(diag) if c in x.keys()]
    if diag + "_extra" in x.keys():
        children.append(diag + "_extra")
    if len(children) == 0:
        continue

    z_up = sum(z_bigup[c] for c in children)
    var_up = sum(var_bigup[c] for c in children)

    z_bigup[diag], var_bigup[diag] = combine_estimates(x[diag], var[diag], z_up, var_up)
    n_dead[diag] = sum(n_dead[c] for c in children)

x_hat, var_hat = OrderedDict({}), OrderedDict({})
z_bigdown, var_bigdown = OrderedDict({}), OrderedDict({})
for diag in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "19", "NODIAG"]:
    x_hat[diag] = z_bigup[diag]
    var_hat[diag] = var_bigup[diag]
    z_bigdown[diag] = x[diag]
    var_bigdown[diag] = var[diag]

for diag in sorted(x.keys(), key=lambda k: find_depth(k)):
    if len(diag) == 2 or diag == "NODIAG":
        continue

    p = find_parent(diag)
    siblings = set(find_children(p)) - {diag} | {p + "_extra"}
    z_down = z_bigdown[p] - sum(z_bigup.get(s, 0) for s in siblings)
    var_down = var_bigdown[p] + sum(var_bigup.get(s, 0) for s in siblings)

    x_hat[diag], var_hat[diag] = combine_estimates(z_bigup[diag], var_bigup[diag], z_down, var_down)
    z_bigdown[diag], var_bigdown[diag] = combine_estimates(x[diag], var[diag], z_down, var_down)

df_diag_final = (pl
    .from_dicts({
        "diagnosis": x_hat.keys(),
        "n_dead": [n_dead[k] for k in x_hat.keys()],
        "n_alive": [x_hat[k] for k in x_hat.keys()],
        "n_alive_var": [var_hat[k] for k in x_hat.keys()],
    })
    .with_columns(
        pl.when(pl.col("n_dead") == 0, pl.col("n_alive") < 1.)
          .then(1.)
          .otherwise(
              pl.col("n_alive").clip(0.)
          )
          .alias("n_alive")
    )
    .with_columns(
        (pl.col("n_dead") + pl.col("n_alive")).alias("n_pop"),
        pl.col("diagnosis").map_elements(lambda x: find_all_parents(x), return_dtype=pl.List(pl.String)).alias("diag_parents")
    )
)

# English term of each diagnosis
icd_se = pl.read_excel("mappings/missing_terms_JS.xlsx")
icd = (pl
    .concat([
        load_icd().filter(~pl.col("code").is_in(icd_se["code"])),
        pl.DataFrame({
            "code": ["NODIAG"],
            "term": [""],
            "term_en": ["The patient received no diagnosis within 30 days of ED visit"]
        }),
        icd_se
    ])
)

# Beta prior of the mortality risk across diagnoses (empirical Bayes)
with pm.Model() as model:
    ab = pm.HalfFlat("ab", shape=2)
    pm.Potential("p(a, b)", logp_ab(ab))
    alpha = pm.Deterministic("alpha", ab[0])
    beta = pm.Deterministic("beta", ab[1])
    theta = pm.Beta("theta", alpha=ab[0], beta=ab[1], shape=len(df_diag_final))
    pm.Binomial(
        "y",
        p=theta,
        observed=df_diag_final["n_dead"].to_numpy(),
        n=df_diag_final["n_pop"].to_numpy()
    )
    ns = pm.Deterministic("ns", alpha + beta)
    mu = pm.Deterministic("mu", alpha / ns)

    idata = pm.sample(random_seed=0)
print(az.summary(idata, ["mu", "ns"]))

# Hierarchical model: the prior of a diagnosis depends on its ICD ancestors
mlb = MultiLabelBinarizer()
diag_dummies = mlb.fit_transform(
    df_diag_final.select("diag_parents").to_series()
)
df_dummies = pl.DataFrame(diag_dummies, [(c, pl.Boolean) for c in mlb.classes_])
df_diag_full = pl.concat([
    df_diag_final,
    df_dummies.select(pl.all().exclude("^.*_extra$"))
], how="horizontal").filter(~pl.col("diagnosis").str.ends_with("_extra"))

W = df_diag_full.select(cs.boolean()).to_numpy()
W = np.hstack([np.ones([W.shape[0], 1]), W])
n, d = W.shape

with pm.Model() as model:
    ab = pm.HalfNormal("ab", sigma=3. / W.sum(axis=1).mean(), shape=(2, d))
    alpha = pm.Deterministic("alpha", W @ ab[0])
    beta = pm.Deterministic("beta", W @ ab[1])
    theta = pm.Beta("theta", alpha=alpha, beta=beta)
    pm.Binomial(
        "y",
        p=theta,
        observed=df_diag_full["n_dead"].to_numpy(),
        n=df_diag_full["n_pop"].to_numpy()
    )
    ns = pm.Deterministic("ns", alpha + beta)
    mu = pm.Deterministic("mu", alpha / ns)

    idata = pm.sample(random_seed=0)

df_diag_hier = pl.concat([
    df_diag_full
        .select("diagnosis", "n_pop", "n_dead")
        .with_columns(
            pl.col("diagnosis")
              .replace_strict(
                  old=icd["code"],
                  new=icd["term_en"],
                  default=None
              )
              .alias("term")
        ),
    pl.DataFrame({
        "mu": np.maximum(idata
            .posterior.theta
            .to_numpy()
            .reshape(4000, -1)
            .mean(0)
            .round(4),
            1e-4
        )
    })
], how="horizontal")

diagnoses = json.dumps(
    df_diag_hier.rename({"diagnosis": "id"}).to_dicts(),
    separators=(",", ":")
)
(out / "diagnoses.json").write_text(diagnoses)


# *** MORTALITY MODEL ***
n_digits = 4

df_all = pl.read_csv("models/static_mort_predictor.csv", infer_schema_length=None)
static_model = {}
for symptom in df_all["symptom"].unique():
    df_sub = df_all.filter(pl.col("symptom") == symptom)
    df_male = df_sub.filter(pl.col("is_male"))
    df_female = df_sub.filter(~pl.col("is_male"))
    static_model[symptom] = {
        "M": {
            "mu": df_male["mu"].round(n_digits).to_list(),
            "sig": df_male["sigma"].round(n_digits).to_list()
        },
        "F": {
            "mu": df_female["mu"].round(n_digits).to_list(),
            "sig": df_female["sigma"].round(n_digits).to_list()
        },
    }
(out / "static_model.json").write_text(json.dumps(static_model, separators=(",", ":")))

df_all_base = pl.read_csv("models/static_base_mort_predictor.csv", infer_schema_length=None)
df_male = df_all_base.select("is_male", "ed_age", "mu", "sigma").filter(pl.col("is_male"))
df_female = df_all_base.select("is_male", "ed_age", "mu", "sigma").filter(~pl.col("is_male"))
static_base_model = {
    "M": {
        "mu": df_male["mu"].round(n_digits).to_list(),
        "sig": df_male["sigma"].round(n_digits).to_list()
    },
    "F": {
        "mu": df_female["mu"].round(n_digits).to_list(),
        "sig": df_female["sigma"].round(n_digits).to_list()
    },
}
(out / "static_base_model.json").write_text(json.dumps(static_base_model, separators=(",", ":")))


# *** INDEXES ***
df = pl.read_parquet("datasets/diagnoses_dummies.parquet")
all_diagnoses_str = sorted(df.select(cs.boolean().exclude("is_male")).columns)
(load_icd()
    .filter(pl.col("code").is_in(all_diagnoses_str))
    .select("code", pl.col("term_en").alias("term"))
    .write_json(out / "index_diagnoses.json")
)

# Most common symptoms first, with the "no symptom" entry last
symptoms = (df_mort
    .with_columns(pl.col("symptoms").str.strip_chars("[]"))
    .group_by("symptoms")
    .agg(count=pl.col("count").sum())
    .sort("count", descending=True)
    .select("symptoms")
)
symptoms = symptoms.slice(1, symptoms.height - 1).vstack(symptoms.slice(0, 1))

df2 = pl.read_parquet("datasets/diagnoses_parents.parquet")
(symptoms
    .join(
        df2
          .select("symptoms")
          .unique()
          .with_columns(pl.lit(True).alias("is_in_diags")),
        on="symptoms",
        how="left",
    )
    .with_columns(
        pl.col("is_in_diags").fill_null(False)
    )
    .join(snomedct, left_on="symptoms", right_on="code", how="left")
    .rename({"symptoms": "code"})
    .with_columns(pl.col("code").str.replace(":", "_"))
    .write_json(out / "index_symptoms.json")
)
