import polars as pl
from sklearn.preprocessing import MultiLabelBinarizer

print("Loading dataset...")
harmonized = pl.read_excel(
    "mappings/MIMIC-IV_symptoms_mapped.xlsx",
    columns=["chief_complaint", "SNOMED-CT"]
).rename({"chief_complaint": "symptoms"})
patients = pl.read_csv("data/mimic/mimiciv/3.1/hosp/patients.csv.gz")
edstays = (pl
    .read_csv("data/mimic/mimic-iv-ed/2.2/ed/edstays.csv.gz")
    .with_columns(
        pl.col("intime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
    )
)
triage = pl.read_csv("data/mimic/mimic-iv-ed/2.2/ed/triage.csv.gz")
symptoms = (pl
    .read_csv(
        "datasets/mimic_symptoms.txt",
        has_header=False,
        new_columns=["symptoms"],
        separator="\t"
    )
    .with_columns(
        pl.col("symptoms").str.split(", ")
    )
)
df_core = (pl
    .concat([triage, symptoms], how="horizontal")
    .select("subject_id", "stay_id", "symptoms")
    .join(
        patients.select("subject_id", "gender", "dod", pl.col("anchor_age").alias("ed_age")),
        on="subject_id"
    )
    .join(
        edstays.select("stay_id", "intime", "outtime"),
        how="right",
        on="stay_id"
    )
    .with_columns(
        (pl.col("intime") + pl.duration(days=30)).alias("intime_plus_30d"),
        (pl.col("gender") == "M").alias("is_male")
    )
    .with_columns(
        pl.col("dod")
          .is_between(pl.col("intime"), pl.col("intime_plus_30d"))
          .fill_null(False)
          .alias("mort_30d")
    )
)

df_mort = (df_core
    .explode("symptoms")
    .join(harmonized, on="symptoms", how="right")
    .with_columns(pl.col("SNOMED-CT").alias("symptoms"))
    .select(
        "ed_age", "is_male", "mort_30d", "symptoms"
    )
    .with_columns(
        pl.col("symptoms").fill_null("nosct:other")
    )
    .group_by(
        pl.all().exclude("mort_30d")
    )
    .agg(
        count=pl.len(),
        mean_mort30d=pl.col("mort_30d").mean()
    )
)
df_mort.write_parquet("datasets/mimic_mortality.parquet", compression="lz4")
print("*** Generating mortality dataset with dummies")
df_mort_dummies = df_mort.to_dummies("symptoms")
print("Saving...")
df_mort_dummies.write_parquet("datasets/mimic_mortality_dummies.parquet", compression="lz4")

print("*** Generating diagnosis dataset")
admissions = (pl
    .read_csv("data/mimic/mimiciv/3.1/hosp/admissions.csv.gz")
    .select("subject_id", "hadm_id", "dischtime")
)
df_pat = (pl
    .read_csv("data/mimic/mimiciv/3.1/hosp/diagnoses_icd.csv.gz")
    .filter(
        pl.col("icd_version") == 10,
        ~pl.col("icd_code").str.contains(r"^R|^Z")
    )
    .with_columns(
        pl.col("icd_code").alias("diagnosis")
    )
    .join(
        admissions,
        on="hadm_id",
        how="left"
    )
)

df_core_expl = (df_core
    .with_row_index()
    .explode("symptoms")
    .join(harmonized, on="symptoms", how="right")
    .with_columns(pl.col("SNOMED-CT").alias("symptoms"))
    .with_columns(
        pl.col("symptoms").cast(pl.String).fill_null("nosct:other"),
        pl.col("intime").dt.year().alias("year_of_visit")
    )
)

df_diag = (df_core_expl
    .join(
        df_pat,
        on="subject_id",
        how="left"
    )
    .with_columns(
        pl.col("dischtime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S").alias("diag_date")
    )
    .filter(
        pl.col("diag_date")
          .is_between("intime", "intime_plus_30d")
          .fill_null(False)
    )
    .select(
        "index",
        "subject_id",
        "is_male",
        "ed_age",
        "symptoms",
        "year_of_visit",
        pl.col("diagnosis").str.head(4)
    )
    .group_by(
        "index",
        "subject_id",
        "is_male",
        "ed_age",
        "year_of_visit",
        "symptoms"
    )
    .agg(
        diagnosis=pl.col("diagnosis").unique()
    )
)
df_nodiag = (df_core_expl
    .select("index", "subject_id", "is_male", "ed_age", "year_of_visit", "symptoms")
    .join(df_diag, on="index", how="anti")
    .with_columns(diagnosis=pl.lit([], dtype=pl.List(str)))
)

df_total = (
    pl.concat(
        [df_diag, df_nodiag]
    )
    .explode("diagnosis")
    .group_by("ed_age", "is_male", "symptoms", "diagnosis")
    .agg(
        count=pl.len(),
    )
    .with_columns(
        pl.col("symptoms").str.strip_chars("[]"),
        pl.when(pl.col("diagnosis").str.len_chars() == 5)
          .then(pl.col("diagnosis").str.slice(0, 4))
          .otherwise("diagnosis")
    )
)

df_total.write_parquet(
    "datasets/mimic_diagnoses.parquet",
    compression="lz4"
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

print("*** Generating diagnosis dataset with parents...")
# Create a dictionary for quick lookup of parents
parent_dict = dict(zip(icd2parent["code"], icd2parent["parent"]))

# Function to recursively find parents
def find_parents(code, parent_dict):
    parents = [code]
    while code in parent_dict and parent_dict[code] is not None:
        parent = parent_dict[code]
        parents.append(parent)
        code = parent
    return parents

# Add a new column with the list of parents
df_total = (df_total
    .with_columns(
        pl.col("diagnosis")
          .map_elements(
              lambda x: find_parents(x, parent_dict),
              return_dtype=pl.List(pl.String)
          )
          .alias("diag_parents")
    )
)

print("*** Generating diagnosis dataset with dummies...")
mlb = MultiLabelBinarizer()
diag_dummies = mlb.fit_transform(
    df_total.select(
        pl.col("diag_parents").list.tail(1).fill_null(["no_diag"])
    ).to_series()
)
df_dummies = pl.DataFrame(diag_dummies, [(c, pl.Boolean) for c in mlb.classes_])
df_total_dummies = pl.concat([
    df_total,
    df_dummies
], how="horizontal")
print("Saving...")
df_total_dummies.write_parquet(
    "datasets/mimic_diagnoses_dummies.parquet",
    compression="lz4"
)
