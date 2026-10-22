from datetime import date
import time
import polars as pl
from sklearn.preprocessing import MultiLabelBinarizer

# Mock patient used for testing, excluded from every dataset.
MOCK_PATIENT_ID = 701403

prune_diag = True
minimum_leaf_size = 100

start = time.time()

# ICD
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

# Create a dictionary for quick lookup of parents
parent_dict = dict(zip(icd2parent["code"], icd2parent["parent"]))

# Function to recursively find parents
def find_parents(code, parent_dict):
    parents = []
    # While a valid (!=null) parent is found
    while code in parent_dict and parent_dict[code] is not None:
        parent = parent_dict[code]
        parents.append(parent)
        code = parent

    # If we didn't find parents, we could look at a shorter version of the code.
    # There are no parents with a code shorter than 2 though.
    if len(parents) == 0 and len(code) > 2:
        parents = find_parents(code[:-1], parent_dict)

    return parents

# Actual code
df_death = (pl
    .scan_parquet("data/swed/sos_deathreg_wide.parquet")
    .filter(pl.col("lopnr_swed") != MOCK_PATIENT_ID)
)
df_complaint_snomed = pl.read_csv(
    "mappings/presenting_complaint_to_snomed.tsv",
    separator="\t"
)

df_core = (pl
    .scan_parquet("data/swed/swed_ed_core.parquet")
    .filter(pl.col("lopnr_swed") != MOCK_PATIENT_ID)
    .join(df_death, on="lopnr_swed", how="left")
    .with_columns(
        pl.col("ed_age").cast(pl.UInt32),
        pl.col("is_male").cast(pl.Boolean),
        # Date of arrival to ED
        pl.col("ed_arrival_datetime")
          .cast(date)
          .alias("ed_arrival_date")
    )
    .with_columns(
        # 1 month after ED visit
        (pl.col("ed_arrival_date") + pl.duration(days=30))
          .alias("ed_arrival_date + 30")
    )
    .with_columns(
        # Did the patient die within 30 days?
        pl.col("date_death_meanimp")
          .is_between("ed_arrival_date", "ed_arrival_date + 30")
          .fill_null(False)
          .alias("mort_30d"),
        # Presenting complaint
        pl.when(
            pl.col("ed_presented_complaint_nonstd")
              .is_in(df_complaint_snomed.get_column("ed_presented_complaint_nonstd"))
          )
          # Replacing the codes with a harmonized version
          .then(
            pl.col("ed_presented_complaint_nonstd")
              .replace(
                  old=df_complaint_snomed.get_column("ed_presented_complaint_nonstd"),
                  new=df_complaint_snomed.get_column("NEW_ed_presented_complaint_std")
              )
          )
          .otherwise(
            pl.col("ed_presented_complaint_std")
          )
          # Extracting the snomed-ct codes
          .str.extract_all(
            r"\[\d*\]"
          )
          # For null lines
          .fill_null(["nosct:other"])
          .alias("symptoms"),
    )
    .with_columns(
        # Empty lists are populated
        pl.when(pl.col("symptoms").list.len() == 0)
          .then(["nosct:other"])
          .otherwise("symptoms")
          .alias("symptoms")
    )
)

print("*** Generating mortality dataset")
df_mort = (df_core
    .select(["ed_age", "is_male", "mort_30d", "symptoms"])
    .explode("symptoms")
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
df_mort.write_parquet("datasets/mortality.parquet", compression="lz4")

print("*** Generating mortality dataset with dummies")
df_mort_dummies = df_mort.to_dummies("symptoms")
print("Saving...")
df_mort_dummies.write_parquet("datasets/mortality_dummies.parquet", compression="lz4")

print("*** Generating diagnosis dataset")
df_pat = (pl
    .scan_parquet("data/swed/sos_patreg_wide.parquet")
    .filter(pl.col("lopnr_swed") != MOCK_PATIENT_ID)
    .rename({"Hdia": "diagnosis"})
    .filter(
        ~pl.col("diagnosis").str.contains("^R|^Z")
    )
    .with_columns(
        pl.when(pl.col("UTDATUM").is_null())
          .then("INDATUM")
          .otherwise("UTDATUM")
          .alias("diag_date"),
    )
    .select(
        "lopnr_swed",
        "diag_date",
        "diagnosis",
    )
)
def find_lowest_common_ancestor(x, ancestors, parent_dict):
    parents = find_parents(x, parent_dict)
    parents.insert(0, x)
    for parent in parents:
        if parent in ancestors:
            return parent
    return None

ancestors = df_pat.select("diagnosis").unique().collect()
# Maps terms from death registry to censored patient registry
death2pat = (df_death
    .select("ULORSAK")
    .unique("ULORSAK")
    .with_columns(
        pl.col("ULORSAK")
          .map_elements(
              lambda x: find_lowest_common_ancestor(x, ancestors["diagnosis"], parent_dict),
              return_dtype=pl.String
          )
          .alias("lca")
    )
    .rename({
        "ULORSAK": "death_code",
        "lca": "patient_code"
    })
    .drop_nulls()
    .collect()
)
df_pat = pl.concat([
    df_pat
        .with_columns(
            from_registry=pl.lit("patient")
        ),
    df_death
        .select("lopnr_swed", "date_death_meanimp", "ULORSAK")
        .with_columns(
            pl.col("ULORSAK").replace(
                old=death2pat["death_code"],
                new=death2pat["patient_code"]
            ),
            pl.lit("death").alias("from_registry")
        )
        .rename({
            "date_death_meanimp": "diag_date",
            "ULORSAK": "diagnosis"
        })
        .filter(
            ~pl.col("diagnosis").str.contains("^R|^Z")
        )
])

df_core_expl = (df_core
    .filter(pl.col("ed_arrival_date") <= date(2016, 12, 31))
    .with_row_index()
    .explode("symptoms")
)
df_diag = (df_core_expl
    .join(
        df_pat,
        on="lopnr_swed",
        how="left"
    )
    .filter(
        pl.col("diag_date")
          .is_between("ed_arrival_date", "ed_arrival_date + 30")
          .fill_null(False)
    )
    .group_by(
        "index", "lopnr_swed", "ed_age", "is_male", "symptoms", "mort_30d", "from_registry"
    )
    .agg(
        diagnosis=pl.col("diagnosis").unique()
    )
)
df_nodiag = (df_core_expl
    .select(["index", "lopnr_swed", "ed_age", "is_male", "symptoms", "mort_30d"])
    .join(df_diag, on="index", how="anti")
    .with_columns(
        from_registry=pl.lit("patient"),
        diagnosis=pl.lit([], dtype=pl.List(str)),
    )
)
# Translate typos and old terms to the new codes:
typos = pl.read_csv("mappings/obsolete_typos.csv")
df_total = (
    pl.concat(
        [df_diag, df_nodiag]
    )
    .explode("diagnosis")
    .group_by(["ed_age", "is_male", "symptoms", "diagnosis", "from_registry"])
    .agg(
        count=pl.len(),
        mean_mort30d=pl.col("mort_30d").mean()
    )
    .with_columns(
        pl.col("symptoms").str.strip_chars("[]"),
        pl.when(pl.col("diagnosis").str.len_chars() == 5)
          .then(pl.col("diagnosis").str.slice(0, 4))
          .otherwise("diagnosis")
          .replace(old=typos["old_code"], new=typos["new_code"])
          .fill_null("NODIAG")
    )
    .collect()
)

# Creating diagnosis scatter database
print("Creating diagnosis scatter database")
df_diag_unique = (df_core
    .filter(pl.col("ed_arrival_date") <= date(2016, 12, 31))
    .with_row_index()
    .join(
        df_pat.filter(pl.col("from_registry") == "patient"),
        on="lopnr_swed",
        how="left"
    )
    .filter(
        pl.col("diag_date")
          .is_between("ed_arrival_date", "ed_arrival_date + 30")
          .fill_null(False)
    )
    .with_columns(
        pl.when(pl.col("diagnosis").str.len_chars() == 5)
          .then(pl.col("diagnosis").str.slice(0, 4))
          .otherwise("diagnosis")
          .replace(old=typos["old_code"], new=typos["new_code"])
          .fill_null("NODIAG")
    )
    .group_by("index")
    .agg(
        pl.col("diagnosis").unique(),
        pl.col("mort_30d").first()
    )
    .explode("diagnosis")
    .group_by("diagnosis")
    .agg(
        n_dead=pl.col("mort_30d").sum(),
        n_pop=pl.len()
    )
    .sort("diagnosis")
).collect().write_parquet("datasets/mortality_unique.parquet")

if prune_diag:
    prev_leaves_to_merge = None
    while True:
        # Computes the weighted value_counts
        leaf_sizes = df_total.group_by("diagnosis").agg(count=pl.col("count").sum())
        leaves_to_merge = (leaf_sizes
            # If the leaves are smaller than the threshold
            .filter(pl.col("count") < minimum_leaf_size)
            # And they have a parent in the mapping
            .filter(pl.col("diagnosis").is_in(icd2parent["code"]))
            .with_columns(
                pl.col("diagnosis")
                  .replace(old=icd2parent["code"], new=icd2parent["parent"])
                  .alias("parent")
            )
        )
        print(f"There are {len(leaf_sizes)} unique codes at the moment")
        if (
                prev_leaves_to_merge is not None and
                leaves_to_merge["diagnosis"].is_in(prev_leaves_to_merge["diagnosis"]).all()
           ):
            print("Reached idempotence, will stop merging.")
            break
        else:
            print(f"Found {len(leaves_to_merge)} leaves to merge with parent!")
            prev_leaves_to_merge = leaves_to_merge.clone()

        df_total = (df_total
            .with_columns(
                pl.col("diagnosis")
                  .replace(old=leaves_to_merge["diagnosis"], new=leaves_to_merge["parent"])
            )
        )
    # Leaves that couldn't be merged or failed to gain enough mass are discarded
    leaves_to_discard = (leaf_sizes
        .filter(pl.col("count") < minimum_leaf_size)
        .get_column("diagnosis")
    )
    df_total = (df_total
        .filter(~pl.col("diagnosis").is_in(leaves_to_discard).fill_null(False))
    )

if prune_diag:
    print("Saving...")
    df_total.write_parquet("datasets/diagnoses_pruned.parquet", compression="lz4")
else:
    print("Skipped pruning!")
    print("Saving...")
    df_total.write_parquet("datasets/diagnoses.parquet", compression="lz4")

print("*** Generating diagnosis dataset with parents...")
# Add a new column with the list of parents
df_total = (df_total
    .with_columns(
        pl.col("diagnosis")
          .map_elements(
              lambda x: find_parents(x, parent_dict)[::-1],
              return_dtype=pl.List(pl.String)
          )
          .list.concat(
              pl.col("diagnosis")
          )
          .alias("diag_parents")
    )
)
df_total.write_parquet(
    "datasets/diagnoses_parents_pruned.parquet" if prune_diag else "datasets/diagnoses_parents.parquet",
    compression="lz4"
)

print("*** Generating diagnosis dataset with dummies...")
mlb = MultiLabelBinarizer()
diag_dummies = mlb.fit_transform(
    df_total.select(
        pl.col("diag_parents").fill_null(["no_diag"])
    ).to_series()
)
df_dummies = pl.DataFrame(diag_dummies, [(c, pl.Boolean) for c in mlb.classes_])
print(f"{df_dummies.shape=}")
df_total_dummies = pl.concat([
    df_total,
    df_dummies
], how="horizontal")
print("Saving...")
df_total_dummies.write_parquet(
    "datasets/diagnoses_dummies_pruned.parquet" if prune_diag else "datasets/diagnoses_dummies.parquet",
    compression="lz4"
)

elapsed = time.time() - start
print(f"Execution: {elapsed:.3f}s")

