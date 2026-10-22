import os
import polars as pl

def load_snomed_se(mappings_path="mappings", use_cache=True):
    snomed = (pl
        .read_csv(mappings_path + "/sct2_Description_Full-sv_SE1000052_20230531.txt", separator="\t")
        .sort("effectiveTime")
        .group_by("conceptId")
        .agg(pl.col("effectiveTime", "term").last())
        .rename({"conceptId": "code"})
        .select(pl.all().exclude("effectiveTime"))
    )
    snomed = pl.concat([
        snomed,
        pl.DataFrame({"code": "nosct:other", "term": "Igen SNOMED-CT-kartläggning"})
    ])
    return snomed

def load_snomed_en(mappings_path="mappings", use_cache=True):
    if not os.path.exists(mappings_path + "/snomed_en.parquet") or not use_cache:
        print("SNOMED dataset not generated! I will do it now, then!")
        refset = pl.concat([
            pl.read_csv(mappings_path + "/der2_cRefset_LanguageFull-en_INT_20230531.txt", separator="\t"),
            pl.read_csv(mappings_path + "/der2_cRefset_LanguageFull-en_SE1000052_20230531.txt", separator="\t")
        ])
        snomed = (pl
            .concat([
                pl.read_csv(mappings_path + "/sct2_Description_Full-en_INT_20230531.txt", separator="\t"),
                pl.read_csv(mappings_path + "/sct2_Description_Full-en_SE1000052_20230531.txt", separator="\t")
            ])
            .join(
                refset.select("referencedComponentId", "acceptabilityId", "refsetId"),
                left_on="id",
                right_on="referencedComponentId"
            )
            .filter(
                pl.col("acceptabilityId") == 900000000000548007, # Preferred
                pl.col("refsetId") == 900000000000509007, # United States of America English language reference set
                pl.col("typeId") == 900000000000013009, # Synonym
            )
            .unique("conceptId")
            .select(pl.col("conceptId").cast(pl.String), "term")
            .rename({"conceptId": "code"})
        )
        snomed.write_parquet(mappings_path + "/snomed_en.parquet")
    else:
        snomed = pl.read_parquet(mappings_path + "/snomed_en.parquet")

    snomed = pl.concat([
        snomed,
        pl.DataFrame({"code": "nosct:other", "term": "No SNOMED-CT mapping"})
    ])
    return snomed

