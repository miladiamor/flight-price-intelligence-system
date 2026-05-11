#!/usr/bin/env python3
"""
Fill continent and country fields from routes reference data.

Reads:
    data/final/master_training_dataset_full_routes.csv
    data/reference/routes_features_final_continents_filled.csv

Fills:
    origin_iso_country
    dest_iso_country
    origin_continent
    dest_continent
    continent_pair

Writes:
    data/final/master_training_dataset_enriched_final.csv
"""

from pathlib import Path
import os
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # cheapflight-scraper-main/

FINAL_DIR = BASE_DIR / "data" / "final"
REF_DIR = BASE_DIR / "data" / "reference"

MASTER_PATH = FINAL_DIR / "master_training_dataset_full_routes.csv"
ROUTES_PATH = REF_DIR / "routes_features_final_continents_filled.csv"

OUTPUT_PATH = FINAL_DIR / "master_training_dataset_enriched_final.csv"

CHUNK_SIZE = 200_000


COLS_TO_FILL = [
    "origin_iso_country",
    "dest_iso_country",
    "origin_continent",
    "dest_continent",
    "continent_pair",
]


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    return pd.read_csv(path, low_memory=False)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)[:50]}"
    )


def load_routes_lookup() -> pd.DataFrame:
    print("Loading routes lookup...")
    routes = read_table(ROUTES_PATH)

    origin_col = find_column(
        routes,
        ["origin_iata", "origin_iata_code", "origin_iata_co", "Origin", "origin"]
    )

    dest_col = find_column(
        routes,
        ["destination_iata", "dest_iata_code", "dest_iata_co", "Destination", "destination"]
    )

    keep_cols = [origin_col, dest_col] + [c for c in COLS_TO_FILL if c in routes.columns]

    routes = routes[keep_cols].copy()

    routes["origin_iata"] = (
        routes[origin_col]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    routes["destination_iata"] = (
        routes[dest_col]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    agg_dict = {c: "first" for c in COLS_TO_FILL if c in routes.columns}

    routes_lookup = (
        routes.groupby(["origin_iata", "destination_iata"], as_index=False)
        .agg(agg_dict)
    )

    print("Routes lookup rows:", len(routes_lookup))

    return routes_lookup


def fill_chunk(chunk: pd.DataFrame, routes_lookup: pd.DataFrame) -> pd.DataFrame:
    for col in ["origin_iata", "destination_iata"]:
        if col not in chunk.columns:
            raise KeyError(f"Master file missing required column: {col}")

        chunk[col] = (
            chunk[col]
            .astype("string")
            .str.upper()
            .str.strip()
        )

    merged = chunk.merge(
        routes_lookup,
        on=["origin_iata", "destination_iata"],
        how="left",
        suffixes=("", "_rt"),
    )

    for base in COLS_TO_FILL:
        rt = base + "_rt"

        if rt not in merged.columns:
            continue

        if base not in merged.columns:
            merged[base] = merged[rt]
        else:
            mask = merged[base].isna() | (
                merged[base].astype("string").str.strip() == ""
            )
            merged.loc[mask, base] = merged.loc[mask, rt]

        merged.drop(columns=[rt], inplace=True)

    if "origin_continent" in merged.columns and "dest_continent" in merged.columns:
        oc = merged["origin_continent"].astype("string").str.strip()
        dc = merged["dest_continent"].astype("string").str.strip()

        mask = (
            oc.notna()
            & dc.notna()
            & (oc != "")
            & (dc != "")
            & (oc != "nan")
            & (dc != "nan")
        )

        merged.loc[mask, "continent_pair"] = oc[mask] + "_" + dc[mask]

    return merged


def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master file: {MASTER_PATH}")

    routes_lookup = load_routes_lookup()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        os.remove(OUTPUT_PATH)

    print("Processing master file in chunks:")
    print(MASTER_PATH)

    first_chunk = True
    total_rows = 0

    for i, chunk in enumerate(
        pd.read_csv(MASTER_PATH, chunksize=CHUNK_SIZE, low_memory=False),
        start=1
    ):
        print(f"Chunk {i}: {len(chunk)} rows")

        filled = fill_chunk(chunk, routes_lookup)
        total_rows += len(filled)

        filled.to_csv(
            OUTPUT_PATH,
            mode="a",
            index=False,
            header=first_chunk,
        )

        first_chunk = False

    print("\nDone.")
    print("Rows written:", total_rows)
    print("Saved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()