#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT = BASE_DIR / "data" / "raw" / "flights_step1_dedup.csv"
OUTPUT = BASE_DIR / "data" / "raw" / "flights_step2_clean.csv"


# -----------------------------
# Helpers
# -----------------------------
def clean_price(x):
    """
    Convert:
      €329
      USD 1,240
      £540
    -> float
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    # keep digits only
    x = re.sub(r"[^0-9]", "", x)

    if x == "":
        return np.nan

    return float(x)


def clean_duration(x):
    """
    Convert:
      5 hr 30 min -> 330
      2 hr -> 120
    """
    if pd.isna(x):
        return np.nan

    x = str(x)

    h = re.search(r"(\d+)\s*hr", x)
    m = re.search(r"(\d+)\s*min", x)

    hours = int(h.group(1)) if h else 0
    mins = int(m.group(1)) if m else 0

    return hours * 60 + mins


def normalize_stops(x):
    """
    Convert:
      Nonstop -> 0
      1 stop -> 1
      2 stops -> 2
    """
    if pd.isna(x):
        return np.nan

    x = str(x).lower()

    if "nonstop" in x:
        return 0

    m = re.search(r"(\d+)", x)
    return int(m.group(1)) if m else np.nan


# -----------------------------
# Main
# -----------------------------
def main():

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)

    print("Cleaning prices...")
    df["Price_num"] = df["Price"].apply(clean_price)

    print("Cleaning duration...")
    df["Duration_min"] = df["Duration"].apply(clean_duration)

    print("Cleaning stops...")
    df["Stops_num"] = df["Stops"].apply(normalize_stops)

    print("Converting dates...")

    # Flight departure date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Search date from scraper
    if "Search Date" in df.columns:
        df["Search Date"] = pd.to_datetime(df["Search Date"], errors="coerce")

    print("Saving cleaned file...")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT, index=False)

    print(f"Saved → {OUTPUT}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()