#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "raw" / "flights.csv"
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "flights_step1_dedup.csv"


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print("Rows before dedup:", len(df))

    # remove fully identical rows
    df_dedup = df.drop_duplicates(keep="first")

    print("Rows after dedup :", len(df_dedup))
    print("Duplicates removed:", len(df) - len(df_dedup))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df_dedup.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved de-duplicated data to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()