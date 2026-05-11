#!/usr/bin/env python3
"""
Feature Step 3: Add stop-type flags.

Reads:
    data/final/flights_features_step2_full.csv

Adds:
    Is_nonstop
    Is_onestop
    Is_multistop

Writes:
    data/final/flights_features_step3_stops.csv
"""

from pathlib import Path
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "final"
    / "flights_features_step2_full.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "final"
    / "flights_features_step3_stops.csv"
)


def main():

    # -----------------------------
    # Check input
    # -----------------------------
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded: {INPUT_FILE}")
    print(f"Rows: {len(df)}")

    # -----------------------------
    # Validate Stops_num
    # -----------------------------
    if "Stops_num" not in df.columns:
        raise ValueError(
            "Expected 'Stops_num' column. "
            "Run flights2_features_step2_full.py first."
        )

    # -----------------------------
    # Clean Stops_num
    # -----------------------------
    df["Stops_num"] = (
        pd.to_numeric(df["Stops_num"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # -----------------------------
    # Create stop flags
    # -----------------------------
    df["Is_nonstop"] = (df["Stops_num"] == 0).astype(int)

    df["Is_onestop"] = (df["Stops_num"] == 1).astype(int)

    df["Is_multistop"] = (df["Stops_num"] >= 2).astype(int)

    # -----------------------------
    # Summary
    # -----------------------------
    print("\n=== Stop Flags Summary ===")
    print("Nonstop flights   :", df["Is_nonstop"].sum())
    print("One-stop flights  :", df["Is_onestop"].sum())
    print("Multi-stop flights:", df["Is_multistop"].sum())

    # -----------------------------
    # Save
    # -----------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved -> {OUTPUT_FILE}")
    print("Final rows:", len(df))


if __name__ == "__main__":
    main()