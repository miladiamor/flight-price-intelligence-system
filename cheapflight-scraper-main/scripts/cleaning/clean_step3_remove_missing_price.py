#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT = BASE_DIR / "data" / "raw" / "flights_step2_clean.csv"
OUTPUT = BASE_DIR / "data" / "final" / "flights_clean_full.csv"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)

    if "Price_num" not in df.columns:
        raise ValueError("Price_num column not found. Run clean_step2_types.py first.")

    print("Before deletion:", df.shape)

    df = df[df["Price_num"].notna()]
    df = df[df["Price_num"] > 0]

    print("After deletion:", df.shape)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved cleaned data → {OUTPUT}")


if __name__ == "__main__":
    main()