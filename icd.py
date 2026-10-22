import polars as pl

def load_icd(mappings_path="mappings"):
    all_diagnoses = (pl
        .read_csv(
           mappings_path + "/icd-10-se.tsv",
           columns=["Kod", "Titel"],
           new_columns=["code", "term"],
           separator="\t",
           schema_overrides=pl.Schema({"Kod": pl.String, "Titel": pl.String}),
           null_values=""
        )
        .drop_nulls()
        .with_columns(
           pl.all().str.replace(r"\.", "")
        )
    )
    # English support
    diag_en = pl.concat([
        pl.read_csv(
            mappings_path + "/icd102019syst_chapters.txt",
            separator=";",
            has_header=False,
            new_columns=["code", "term"],
            schema={"code": pl.String, "term": pl.String}
        ),
        pl.read_csv(
            mappings_path + "/icd102019syst_groups.txt",
            separator=";",
            has_header=False,
            new_columns=["begin", "end", "chapter", "term"]
        )
        .with_columns(
            (pl.col("begin") + "-" + pl.col("end")).alias("code")
        )
        .select("code", "term"),
        pl.read_csv(
            mappings_path + "/icd102019syst_codes.txt",
            separator=";",
            has_header=False,
            infer_schema_length=None,
            new_columns= ["level", "terminal", "type", "chapter", "code1", "code2", "code3", "code", "term"]
        ).select("code", "term")
    ])
    all_diagnoses = all_diagnoses.with_columns(
        pl.col("code")
          .replace_strict(old=diag_en["code"], new=diag_en["term"], default="NOTRANS")
          .alias("term_en")
    )
    return all_diagnoses
