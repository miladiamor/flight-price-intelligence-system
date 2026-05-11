#!/usr/bin/env python3
"""
Fill airline codes and normalize stop values.

Reads:
    data/final/master_training_dataset_enriched_airports.csv
    data/reference/airline_codes.csv

Adds/fills:
    airline
    stops

Writes:
    data/final/master_training_dataset_enriched_airlines.csv

Also writes debugging files:
    data/final/unmatched_airlines_rows.csv
    data/final/unmatched_airline_names.txt
"""

from pathlib import Path
import re
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

FINAL_DIR = BASE_DIR / "data" / "final"
REF_DIR = BASE_DIR / "data" / "reference"

MASTER_PATH = FINAL_DIR / "master_training_dataset_enriched_airports.csv"
AIRLINE_CODES_PATH = REF_DIR / "airline_codes.csv"

OUTPUT_PATH = FINAL_DIR / "master_training_dataset_enriched_airlines.csv"

UNMATCHED_ROWS_PATH = FINAL_DIR / "unmatched_airlines_rows.csv"
UNMATCHED_NAMES_PATH = FINAL_DIR / "unmatched_airline_names.txt"

OPENFLIGHTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"


def clean_airline_name(s):
    if pd.isna(s):
        return pd.NA

    s = str(s)
    s = re.split(r"\boperated by\b", s, flags=re.IGNORECASE)[0]
    s = re.split(r"[+,/]", s)[0]

    return s.strip()


def parse_stops_text(val):
    if pd.isna(val):
        return pd.NA

    s = str(val).strip().lower()

    if not s or s == "nan":
        return pd.NA

    if "nonstop" in s or "non-stop" in s:
        return 0

    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else pd.NA


def read_airline_codes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing airline code file: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        codes = pd.read_excel(path, dtype="string")
    else:
        codes = pd.read_csv(path, dtype="string")

    if "airline" not in codes.columns or "airline_name" not in codes.columns:
        raise KeyError(
            "airline_codes file must contain columns: airline, airline_name"
        )

    codes["airline"] = codes["airline"].astype("string").str.upper().str.strip()
    codes["airline_name_clean"] = (
        codes["airline_name"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    codes["source_priority"] = 0

    return codes


def build_name_to_code_mapping() -> dict:
    print("Loading local airline codes...")
    codes = read_airline_codes(AIRLINE_CODES_PATH)

    try:
        print("Trying OpenFlights fallback mapping...")
        of = pd.read_csv(
            OPENFLIGHTS_URL,
            header=None,
            names=[
                "id",
                "name",
                "alias",
                "iata",
                "icao",
                "callsign",
                "country",
                "active",
            ],
        )

        mask_real = of["iata"].notna() & (of["iata"] != r"\N") & (of["iata"] != "")
        of = of.loc[mask_real, ["name", "iata"]].copy()

        of["airline"] = of["iata"].astype("string").str.upper().str.strip()
        of["airline_name_clean"] = (
            of["name"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        of["source_priority"] = 1

        combined = pd.concat(
            [
                codes[["airline", "airline_name_clean", "source_priority"]],
                of[["airline", "airline_name_clean", "source_priority"]],
            ],
            ignore_index=True,
        )

    except Exception as e:
        print("Warning: OpenFlights not available. Using local airline_codes only.")
        print("Reason:", e)

        combined = codes[["airline", "airline_name_clean", "source_priority"]]

    combined = (
        combined.dropna(subset=["airline_name_clean", "airline"])
        .sort_values("source_priority")
        .drop_duplicates(subset=["airline_name_clean"], keep="first")
    )

    mapping = combined.set_index("airline_name_clean")["airline"].to_dict()

    print(f"Built airline mapping for {len(mapping)} names.")
    return mapping


def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master file: {MASTER_PATH}")

    print("Loading master dataset...")
    master = pd.read_csv(MASTER_PATH, low_memory=False)

    print("Rows:", len(master))

    if "Airline" not in master.columns:
        raise ValueError("Master dataset missing required column: Airline")

    # -----------------------------
    # Stops
    # -----------------------------
    print("Normalizing stops...")

    if "Stops_num" in master.columns:
        master["stops"] = pd.to_numeric(master["Stops_num"], errors="coerce")
    else:
        master["stops"] = pd.NA

    if "Stops" in master.columns:
        mask_missing = master["stops"].isna()
        if mask_missing.any():
            master.loc[mask_missing, "stops"] = (
                master.loc[mask_missing, "Stops"].map(parse_stops_text)
            )

    master["stops"] = pd.to_numeric(master["stops"], errors="coerce").astype("Int64")

    # -----------------------------
    # Airline codes
    # -----------------------------
    name_to_code = build_name_to_code_mapping()

    print("Generating airline codes...")

    if "airline" not in master.columns:
        master["airline"] = pd.NA

    master["Airline_clean"] = master["Airline"].astype("string").map(clean_airline_name)
    master["Airline_clean_lower"] = master["Airline_clean"].str.lower()

    master["airline_from_name"] = master["Airline_clean_lower"].map(name_to_code)

    master["airline"] = master["airline_from_name"].fillna(master["airline"])
    master["airline"] = master["airline"].astype("string").str.upper().str.strip()
    master.loc[master["airline"] == "NAN", "airline"] = pd.NA

    # -----------------------------
    # Export unmatched airlines
    # -----------------------------
    print("Checking unmatched airlines...")

    mask_unmatched = master["airline"].isna() | (
        master["airline"].astype("string").str.strip() == ""
    )

    unmatched_rows = master.loc[mask_unmatched].copy()

    if not unmatched_rows.empty:
        unmatched_rows.to_csv(UNMATCHED_ROWS_PATH, index=False)

        unique_names = (
            unmatched_rows["Airline_clean"]
            .dropna()
            .astype("string")
            .str.strip()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        with open(UNMATCHED_NAMES_PATH, "w", encoding="utf-8") as f:
            for name in unique_names:
                f.write(name + "\n")

        print(f"Unmatched rows saved: {UNMATCHED_ROWS_PATH}")
        print(f"Unmatched names saved: {UNMATCHED_NAMES_PATH}")
        print("Unmatched count:", len(unmatched_rows))
    else:
        print("No unmatched airlines.")

    # Drop helper columns
    master = master.drop(
        columns=["Airline_clean", "Airline_clean_lower", "airline_from_name"],
        errors="ignore",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:")
    print(OUTPUT_PATH)
    print("Rows:", len(master))


if __name__ == "__main__":
    main()