#!/usr/bin/env python3
"""
Fill missing route-level features from routes reference data.

Reads:
    data/final/master_training_dataset_enriched_airlines.csv
    data/reference/routes_features_final_continents_filled.csv

Fills:
    missing route metadata using origin_iata + destination_iata

Writes:
    data/final/master_training_dataset_full_routes.csv
"""

from pathlib import Path
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # cheapflight-scraper-main/

FINAL_DIR = BASE_DIR / "data" / "final"
REF_DIR = BASE_DIR / "data" / "reference"

MASTER_PATH = FINAL_DIR / "master_training_dataset_enriched_airlines.csv"
ROUTES_PATH = REF_DIR / "routes_features_final_continents_filled.csv"

OUTPUT_PATH = FINAL_DIR / "master_training_dataset_full_routes.csv"


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


def normalize_iata_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = (
        df[col]
        .astype("string")
        .str.upper()
        .str.strip()
    )
    return df


def prepare_routes(routes: pd.DataFrame) -> pd.DataFrame:
    origin_col = find_column(
        routes,
        ["origin_iata", "origin_iata_code", "origin_iata_co", "Origin", "origin"]
    )

    dest_col = find_column(
        routes,
        ["destination_iata", "dest_iata_code", "dest_iata_co", "Destination", "destination"]
    )

    routes = routes.copy()

    routes["origin_iata"] = routes[origin_col].astype("string").str.upper().str.strip()
    routes["destination_iata"] = routes[dest_col].astype("string").str.upper().str.strip()

    routes = routes.dropna(subset=["origin_iata", "destination_iata"])

    routes = (
        routes.groupby(["origin_iata", "destination_iata"], as_index=False)
        .first()
    )

    print("Routes unique pairs:", len(routes))

    return routes


def fill_from_routes(master: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    for col in ["origin_iata", "destination_iata"]:
        if col not in master.columns:
            raise KeyError(f"Master file missing required column: {col}")
        master = normalize_iata_column(master, col)

    print("Merging master with routes...")

    merged = master.merge(
        routes,
        on=["origin_iata", "destination_iata"],
        how="left",
        suffixes=("", "_rt"),
    )

    print("Rows after merge:", len(merged))

    print("Filling missing values from route reference...")

    for col in list(merged.columns):
        if not col.endswith("_rt"):
            continue

        base = col[:-3]

        if base in merged.columns:
            base_series = merged[base]
            rt_series = merged[col]

            mask = base_series.isna() | (
                base_series.astype("string").str.strip() == ""
            )

            merged.loc[mask, base] = rt_series[mask]

        else:
            merged[base] = merged[col]

        merged.drop(columns=[col], inplace=True)

    return merged


def main():
    print("Loading master dataset...")
    master = read_table(MASTER_PATH)

    print("Loading routes reference...")
    routes = read_table(ROUTES_PATH)

    print("Master rows:", len(master))
    print("Routes rows:", len(routes))

    routes_prepared = prepare_routes(routes)
    filled = fill_from_routes(master, routes_prepared)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:")
    print(OUTPUT_PATH)
    print("Rows:", len(filled))


if __name__ == "__main__":
    main()