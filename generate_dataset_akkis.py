import polars as pl
from sklearn.preprocessing import MultiLabelBinarizer

print("Loading dataset...")
harmonizer = (pl
    .scan_parquet("mappings/complaint2snomed.parquet")
    .with_columns(
        pl.col("ED_PresentedComplaint")
    )
)

df_core = (pl
    .scan_parquet("data/uppsala/edvisit.parquet")
    .with_columns(
        pl.col("ED_PresentedComplaint")
          .cast(pl.String)
          .replace(
              old="Ögonskada/synpåverkan",
              new="Synpåverkan/Ögonskada"
          ),
        # 1 month after ED visit
        (pl.col("ED_VisitDateTime") + pl.duration(days=30))
          .alias("ED_VisitDateTime + 30"),
    )
    .join(harmonizer, on="ED_PresentedComplaint", how="left")
    .rename({
        "AgeAtED_Visit": "ed_age",
        "SCT-id": "symptoms",
    })
    .filter(
        pl.col("ed_age") >= 18, # excluding -1 meaning unknown
        pl.col("Sex") != "U"
    )
    .with_columns(
        (pl.col("Sex") == "M").alias("is_male"),
        pl.col("ED_VisitDateTime").dt.year().alias("year_of_visit"),
        # Did the patient die within 30 days?
        pl.col("DateOfDeath")
          .is_between("ED_VisitDateTime", "ED_VisitDateTime + 30")
          .fill_null(False)
          .alias("mort_30d"),
    )
)

df_pat = (pl
    .scan_parquet("data/uppsala/healthcarevisits.parquet")
    .rename({"MainDiagnosis": "diagnosis"})
    .with_columns(
        pl.col("diagnosis").cast(pl.String)
    )
    .filter(
        ~pl.col("diagnosis").str.contains("^R|^Z")
    )
    .with_columns(
        pl.when(pl.col("DischargeDate").is_null())
          .then("AdmissionDate")
          .otherwise("DischargeDate")
          .alias("diag_date"),
    )
)

print("*** Generating mortality dataset")
df_mort = (df_core
    .select(["ed_age", "is_male", "mort_30d", "symptoms"])
    .explode("symptoms")
    .with_columns(
        # Same "[code]" format as SwED, so that the dummy columns match those of the mortality model.
        pl.format("[{}]", pl.col("symptoms").cast(pl.String).str.strip_chars("[]")).alias("symptoms")
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
    .collect()
)
print("Saving...")
df_mort.write_parquet("datasets/akkis_mortality.parquet", compression="lz4")

print("*** Generating mortality dataset with dummies")
df_mort_dummies = df_mort.to_dummies("symptoms")
print("Saving...")
df_mort_dummies.write_parquet("datasets/akkis_mortality_dummies.parquet", compression="lz4")

print("*** Generating diagnosis dataset")
df_core_expl = (df_core
    .with_row_index()
    .explode("symptoms")
    .with_columns(
        pl.col("symptoms").cast(pl.String).fill_null("nosct:other")
    )
)
df_diag = (df_core_expl
    .join(
        df_pat,
        on="PersonId",
        how="left"
    )
    .filter(
        pl.col("diag_date")
          .is_between("ED_VisitDateTime", "ED_VisitDateTime + 30")
          .fill_null(False)
    )
    .group_by(
        "index", "year_of_visit", "PersonId", "ed_age", "is_male", "symptoms"
    )
    .agg(
        diagnosis=pl.col("diagnosis").unique()
    )
)
df_nodiag = (df_core_expl
    .select(["index", "year_of_visit", "PersonId", "ed_age", "is_male", "symptoms"])
    .join(df_diag, on="index", how="anti")
    .with_columns(diagnosis=pl.lit([], dtype=pl.List(str)))
)

df_total = (
    pl.concat(
        [df_diag, df_nodiag]
    )
    .explode("diagnosis")
    .group_by(["year_of_visit", "ed_age", "is_male", "symptoms", "diagnosis"])
    .agg(
        count=pl.len(),
    )
    .with_columns(
        pl.col("symptoms").str.strip_chars("[]"),
        pl.when(pl.col("diagnosis").str.len_chars() == 5)
          .then(pl.col("diagnosis").str.slice(0, 4))
          .otherwise("diagnosis")
    )
    .collect()
)

df_total.write_parquet(
    "datasets/akkis_diagnoses.parquet",
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
        pl.col("diag_parents").fill_null(["no_diag"])
    ).to_series()
)
df_dummies = pl.DataFrame(diag_dummies, [(c, pl.Boolean) for c in mlb.classes_])
df_total_dummies = pl.concat([
    df_total,
    df_dummies
], how="horizontal")
print("Saving...")
df_total_dummies.write_parquet(
    "datasets/akkis_diagnoses_dummies.parquet",
    compression="lz4"
)

