#!/usr/bin/env python3
"""
Feature Step 1: Compute Days_left for flights.

Reads:
    data/final/flights_clean_full.csv

Adds:
    Days_left = flight date - search date

Writes:
    data/final/flights_features_step1_daysleft.csv
"""

import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # cheapflight-scraper-main/

INPUT_FILE = BASE_DIR / "data" / "final" / "flights_clean_full.csv"
OUTPUT_FILE = BASE_DIR / "data" / "final" / "flights_features_step1_daysleft.csv"


def robust_parse(series: pd.Series, name: str) -> pd.Series:
    print(f"\n=== Parsing {name} ===")

    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA})

    print("Raw NaN:", s.isna().sum())

    dt1 = pd.to_datetime(s, errors="coerce")
    dt2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    dt3 = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

    combined = dt1.fillna(dt2).fillna(dt3)

    print("Parsed OK:", combined.notna().sum())
    print("Still missing:", combined.isna().sum())

    return combined


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print("Loaded:", INPUT_FILE)
    print("Rows:", len(df))

    if "Date" not in df.columns:
        raise ValueError("Missing required column: Date")

    if "Search Date" not in df.columns:
        raise ValueError("Missing required column: Search Date")

    flight_dt = robust_parse(df["Date"], "Date")
    search_dt = robust_parse(df["Search Date"], "Search Date")

    df["Days_left"] = (flight_dt - search_dt).dt.days

    print("\n=== Days_left summary ===")
    print(df["Days_left"].describe())
    print("Non-missing Days_left:", df["Days_left"].notna().sum())
    print("Still missing:", df["Days_left"].isna().sum())

    bad = (
        df[df["Days_left"].isna()][["Date", "Search Date"]]
        .astype(str)
        .drop_duplicates()
        .head(20)
    )

    if not bad.empty:
        print("\nSample invalid rows:")
        print(bad)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved file ->", OUTPUT_FILE)


if __name__ == "__main__":
    main()