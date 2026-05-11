#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # cheapflight-scraper-main/

RAW_INPUT = BASE_DIR / "data" / "raw" / "flights.csv"

OUT_STEP1 = BASE_DIR / "data" / "raw" / "flights_step1_dedup.csv"
OUT_STEP2 = BASE_DIR / "data" / "raw" / "flights_step2_clean.csv"
OUT_FINAL = BASE_DIR / "data" / "final" / "flights_clean_full.csv"


# --------- helpers ---------
def clean_price(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    x = re.sub(r"[^0-9]", "", x)
    return float(x) if x else np.nan


def clean_duration(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    h = re.search(r"(\d+)\s*hr", x)
    m = re.search(r"(\d+)\s*min", x)
    hours = int(h.group(1)) if h else 0
    mins = int(m.group(1)) if m else 0
    return hours * 60 + mins


def normalize_stops(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower()
    if "nonstop" in x:
        return 0
    m = re.search(r"(\d+)", x)
    return int(m.group(1)) if m else np.nan


# --------- main pipeline ---------
def main():
    if not RAW_INPUT.exists():
        raise FileNotFoundError(f"Raw input not found: {RAW_INPUT}")

    OUT_STEP1.parent.mkdir(parents=True, exist_ok=True)
    OUT_STEP2.parent.mkdir(parents=True, exist_ok=True)
    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)

    # Step 1 — deduplicate
    df = pd.read_csv(RAW_INPUT)
    print("Before dedup:", len(df))

    df = df.drop_duplicates(keep="first")
    print("After  dedup:", len(df))

    df.to_csv(OUT_STEP1, index=False)
    print("Saved step 1:", OUT_STEP1)

    # Step 2 — clean types
    df["Price_num"] = df["Price"].apply(clean_price)
    df["Duration_min"] = df["Duration"].apply(clean_duration)
    df["Stops_num"] = df["Stops"].apply(normalize_stops)

    df.to_csv(OUT_STEP2, index=False)
    print("Saved step 2:", OUT_STEP2)

    # Step 3 — remove missing prices
    before = df.shape
    df = df[df["Price_num"].notna()]
    after = df.shape

    print("Removed:", before[0] - after[0], "rows with missing price")

    df.to_csv(OUT_FINAL, index=False)
    print("Saved final:", OUT_FINAL)


if __name__ == "__main__":
    main()