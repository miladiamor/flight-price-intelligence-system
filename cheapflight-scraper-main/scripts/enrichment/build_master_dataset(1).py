#!/usr/bin/env python3
"""
Build master training dataset.

Reads:
    data/final/flights_features_step3_stops.csv
    data/reference/routes_features_final_continents_filled.csv

Writes:
    data/final/master_training_dataset.csv
    data/final/master_training_dataset_sample100k.csv
"""

from pathlib import Path
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # cheapflight-scraper-main/

FINAL_DIR = BASE_DIR / "data" / "final"
REF_DIR = BASE_DIR / "data" / "reference"

FLIGHTS_PATH = FINAL_DIR / "flights_features_step3_stops.csv"
ROUTES_PATH = REF_DIR / "routes_features_final_continents_filled.csv"

OUTPUT_FULL = FINAL_DIR / "master_training_dataset.csv"
OUTPUT_SAMPLE = FINAL_DIR / "master_training_dataset_sample100k.csv"


def load_flights() -> pd.DataFrame:
    if not FLIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing flights file: {FLIGHTS_PATH}")

    print(f"Loading flights: {FLIGHTS_PATH}")
    flights = pd.read_csv(FLIGHTS_PATH, low_memory=False)

    print("Flights rows before duplicates:", len(flights))
    flights = flights.drop_duplicates()
    print("Flights rows after duplicates :", len(flights))

    required = ["Origin", "Destination"]
    missing = [c for c in required if c not in flights.columns]
    if missing:
        raise ValueError(f"Flights file missing columns: {missing}")

    flights["Origin"] = flights["Origin"].astype(str).str.upper().str.strip()
    flights["Destination"] = flights["Destination"].astype(str).str.upper().str.strip()

    return flights


def load_routes() -> pd.DataFrame:
    if not ROUTES_PATH.exists():
        raise FileNotFoundError(f"Missing routes reference file: {ROUTES_PATH}")

    print(f"Loading routes: {ROUTES_PATH}")
    routes = pd.read_csv(ROUTES_PATH, low_memory=False)

    print("Routes rows raw:", len(routes))

    required = ["origin_iata", "destination_iata"]
    missing = [c for c in required if c not in routes.columns]
    if missing:
        raise ValueError(f"Routes file missing columns: {missing}")

    routes["origin_iata"] = routes["origin_iata"].astype(str).str.upper().str.strip()
    routes["destination_iata"] = routes["destination_iata"].astype(str).str.upper().str.strip()

    routes_agg = (
        routes.groupby(["origin_iata", "destination_iata"], as_index=False)
        .first()
    )

    print("Routes rows aggregated:", len(routes_agg))

    return routes_agg


def build_master_dataset() -> None:
    flights = load_flights()
    routes = load_routes()

    print("\nMerging flights with route metadata...")

    merged = flights.merge(
        routes,
        left_on=["Origin", "Destination"],
        right_on=["origin_iata", "destination_iata"],
        how="left",
        suffixes=("", "_route"),
    )

    print("Rows in flights:", len(flights))
    print("Rows in merged :", len(merged))

    if "distance_km" in merged.columns:
        missing_routes = merged["distance_km"].isna().sum()
        print("Flights without route metadata:", missing_routes)
    else:
        print("Warning: distance_km column not found after merge.")

    OUTPUT_FULL.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(OUTPUT_FULL, index=False)
    print(f"\nSaved full dataset: {OUTPUT_FULL}")

    sample_n = min(100_000, len(merged))
    sample = merged.sample(n=sample_n, random_state=42)

    sample.to_csv(OUTPUT_SAMPLE, index=False)
    print(f"Saved sample dataset: {OUTPUT_SAMPLE}")


if __name__ == "__main__":
    build_master_dataset()