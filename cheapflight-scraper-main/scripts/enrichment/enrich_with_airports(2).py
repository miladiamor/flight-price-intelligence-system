#!/usr/bin/env python3
"""
Enrich master dataset with airport metadata.

Reads:
    data/final/master_training_dataset.csv
    data/reference/global_airports_clean.csv

Adds/fills:
    origin_name
    origin_municipality
    origin_iso_country
    origin_continent
    origin_latitude_deg
    origin_longitude_deg
    origin_type

    dest_name
    dest_municipality
    dest_iso_country
    dest_continent
    dest_latitude_deg
    dest_longitude_deg
    dest_type

Also recalculates:
    direction_lat
    direction_lon

Writes:
    data/final/master_training_dataset_enriched_airports.csv
"""

from pathlib import Path
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # cheapflight-scraper-main/

FINAL_DIR = BASE_DIR / "data" / "final"
REF_DIR = BASE_DIR / "data" / "reference"

MASTER_PATH = FINAL_DIR / "master_training_dataset.csv"
AIRPORTS_PATH = REF_DIR / "global_airports_clean.csv"

OUTPUT_PATH = FINAL_DIR / "master_training_dataset_enriched_airports.csv"


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    return pd.read_csv(path, low_memory=False)


def clean_and_fill_iata(df: pd.DataFrame, target_col: str, source_col: str) -> pd.DataFrame:
    """
    Fill target IATA column from source text column and normalize uppercase.
    """
    if source_col not in df.columns:
        raise ValueError(f"Missing source column: {source_col}")

    if target_col not in df.columns:
        df[target_col] = pd.NA

    df[target_col] = df[target_col].astype("string")
    df[source_col] = df[source_col].astype("string")

    df[target_col] = df[target_col].replace(r"^\s*$", pd.NA, regex=True)
    df[source_col] = df[source_col].replace(r"^\s*$", pd.NA, regex=True)

    df[target_col] = df[target_col].fillna(df[source_col])
    df[target_col] = df[target_col].str.upper().str.strip()

    return df


def prepare_airports(airports: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize airport lookup column names.
    Expected airport file may contain:
        iata_code, name, municipality, iso_country, continent, latitude_deg, longitude_deg, type
    """

    required = ["iata_code"]
    missing = [c for c in required if c not in airports.columns]

    if missing:
        raise ValueError(f"Airports file missing required columns: {missing}")

    airports = airports.copy()

    airports["iata_code"] = (
        airports["iata_code"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    airports = airports.dropna(subset=["iata_code"])
    airports = airports.drop_duplicates(subset=["iata_code"], keep="first")

    return airports


def fill_from_new_columns(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """
    For every base column, fill missing values from base_new column, then drop base_new.
    """
    for base in base_cols:
        new = base + "_new"

        if new not in df.columns:
            continue

        if base not in df.columns:
            df[base] = df[new]
        else:
            df[base] = df[base].fillna(df[new])

        df.drop(columns=[new], inplace=True)

    return df


def main():
    print("Loading master dataset...")
    master = read_table(MASTER_PATH)

    print("Loading airports reference...")
    airports = read_table(AIRPORTS_PATH)
    airports = prepare_airports(airports)

    print("Master rows:", len(master))
    print("Airports rows:", len(airports))

    # Ensure IATA columns exist
    print("Cleaning IATA codes...")
    master = clean_and_fill_iata(master, "origin_iata", "Origin")
    master = clean_and_fill_iata(master, "destination_iata", "Destination")

    if "origin_iata_code" in master.columns:
        master["origin_iata_code"] = (
            master["origin_iata_code"]
            .astype("string")
            .replace(r"^\s*$", pd.NA, regex=True)
            .fillna(master["origin_iata"])
        )

    if "dest_iata_code" in master.columns:
        master["dest_iata_code"] = (
            master["dest_iata_code"]
            .astype("string")
            .replace(r"^\s*$", pd.NA, regex=True)
            .fillna(master["destination_iata"])
        )

    # -----------------------------
    # Origin airport merge
    # -----------------------------
    print("Merging origin airport metadata...")

    origin_air = airports.rename(
        columns={
            "iata_code": "origin_iata",
            "name": "origin_name_new",
            "municipality": "origin_municipality_new",
            "iso_country": "origin_iso_country_new",
            "continent": "origin_continent_new",
            "latitude_deg": "origin_latitude_deg_new",
            "longitude_deg": "origin_longitude_deg_new",
            "type": "origin_type_new",
        }
    )

    master = master.merge(origin_air, on="origin_iata", how="left")

    origin_cols = [
        "origin_name",
        "origin_municipality",
        "origin_iso_country",
        "origin_continent",
        "origin_latitude_deg",
        "origin_longitude_deg",
        "origin_type",
    ]

    master = fill_from_new_columns(master, origin_cols)

    # -----------------------------
    # Destination airport merge
    # -----------------------------
    print("Merging destination airport metadata...")

    dest_air = airports.rename(
        columns={
            "iata_code": "destination_iata",
            "name": "dest_name_new",
            "municipality": "dest_municipality_new",
            "iso_country": "dest_iso_country_new",
            "continent": "dest_continent_new",
            "latitude_deg": "dest_latitude_deg_new",
            "longitude_deg": "dest_longitude_deg_new",
            "type": "dest_type_new",
        }
    )

    master = master.merge(dest_air, on="destination_iata", how="left")

    dest_cols = [
        "dest_name",
        "dest_municipality",
        "dest_iso_country",
        "dest_continent",
        "dest_latitude_deg",
        "dest_longitude_deg",
        "dest_type",
    ]

    master = fill_from_new_columns(master, dest_cols)

    # -----------------------------
    # Direction features
    # -----------------------------
    print("Recalculating direction_lat and direction_lon...")

    if "origin_latitude_deg" in master.columns and "dest_latitude_deg" in master.columns:
        master["origin_latitude_deg"] = pd.to_numeric(master["origin_latitude_deg"], errors="coerce")
        master["dest_latitude_deg"] = pd.to_numeric(master["dest_latitude_deg"], errors="coerce")
        master["direction_lat"] = master["dest_latitude_deg"] - master["origin_latitude_deg"]

    if "origin_longitude_deg" in master.columns and "dest_longitude_deg" in master.columns:
        master["origin_longitude_deg"] = pd.to_numeric(master["origin_longitude_deg"], errors="coerce")
        master["dest_longitude_deg"] = pd.to_numeric(master["dest_longitude_deg"], errors="coerce")
        master["direction_lon"] = master["dest_longitude_deg"] - master["origin_longitude_deg"]

    # -----------------------------
    # Save
    # -----------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved enriched dataset:")
    print(OUTPUT_PATH)
    print("Rows:", len(master))


if __name__ == "__main__":
    main()