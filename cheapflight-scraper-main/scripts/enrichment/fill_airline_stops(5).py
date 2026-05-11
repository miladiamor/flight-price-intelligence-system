#!/usr/bin/env python3
"""
Fill airline codes and normalize stop values.

Reads:
    data/final/master_training_dataset_enriched_airports.csv
    OR fallback:
    data/final/master_training_dataset.csv

    data/reference/airline_codes.csv
    OR fallback:
    data/reference/updated_airline_codes.xlsx
    data/reference/updated_airline_codes.csv

Adds/fills:
    airline
    stops

Writes:
    data/final/master_training_dataset_enriched_airlines.csv

Also writes:
    data/final/unmatched_airlines_rows.csv
    data/final/unmatched_airline_names.txt
"""

from pathlib import Path
import re
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # cheapflight-scraper-main/

FINAL_DIR = BASE_DIR / "data" / "final"
REFERENCE_DIR = BASE_DIR / "data" / "reference"

MASTER_CANDIDATES = [
    FINAL_DIR / "master_training_dataset_enriched_airports.csv",
    FINAL_DIR / "master_training_dataset.csv",
]

AIRLINE_CODE_CANDIDATES = [
    REFERENCE_DIR / "airline_codes.csv",
    REFERENCE_DIR / "updated_airline_codes.xlsx",
    REFERENCE_DIR / "updated_airline_codes.csv",
]

OUTPUT_PATH = FINAL_DIR / "master_training_dataset_enriched_airlines.csv"

UNMATCHED_ROWS_PATH = FINAL_DIR / "unmatched_airlines_rows.csv"
UNMATCHED_NAMES_PATH = FINAL_DIR / "unmatched_airline_names.txt"

OPENFLIGHTS_URL = (
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"
)


def find_existing(path_list: list[Path]) -> Path | None:
    """Return first existing path, else None."""
    for path in path_list:
        if path.exists():
            return path
    return None


def read_table(path: Path, dtype: str | None = None) -> pd.DataFrame:
    """Read CSV or Excel file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, dtype=dtype)

    return pd.read_csv(path, dtype=dtype, low_memory=False)


def clean_airline_name(s: str) -> str:
    """
    Simplify scraped Airline text so it matches airline_codes/OpenFlights.

    Examples:
      'Qatar Airways + JAL' -> 'Qatar Airways'
      'Scandinavian Airlines + Operated by Sas Connect' -> 'Scandinavian Airlines'
      'Air France, WestJet + Operated by HOP!' -> 'Air France'
    """
    if pd.isna(s):
        return pd.NA

    s = str(s)

    # Cut off after 'operated by'
    s = re.split(r"\boperated by\b", s, flags=re.IGNORECASE)[0]

    # Cut at first comma / plus / slash
    s = re.split(r"[+,/]", s)[0]

    return s.strip()


def parse_stops_text(val):
    """Convert 'Nonstop', '1 stop', '2 stops', etc. to an integer."""
    if pd.isna(val):
        return pd.NA

    s = str(val).strip().lower()

    if not s or s == "nan":
        return pd.NA

    if "nonstop" in s or "non-stop" in s:
        return 0

    m = re.search(r"(\d+)", s)

    return int(m.group(1)) if m else pd.NA


def load_airline_codes() -> pd.DataFrame:
    """Load airline code mapping from CSV or Excel."""
    path = find_existing(AIRLINE_CODE_CANDIDATES)

    if path is None:
        raise FileNotFoundError(
            "Could not find any airline code file. Tried:\n"
            + "\n".join(str(p) for p in AIRLINE_CODE_CANDIDATES)
        )

    print(f"Using airline code file: {path}")

    codes = read_table(path, dtype="string")

    if "airline" not in codes.columns or "airline_name" not in codes.columns:
        raise KeyError(
            "Airline code file must contain columns: 'airline' and 'airline_name'. "
            f"Found columns: {list(codes.columns)}"
        )

    codes["airline"] = codes["airline"].astype("string").str.upper().str.strip()
    codes["airline_name_clean"] = (
        codes["airline_name"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    codes["source_priority"] = 0  # local file has highest priority

    return codes


def build_name_to_code_mapping() -> dict:
    """
    Build airline_name_clean -> IATA code mapping from:
      1) local airline_codes file
      2) OpenFlights airlines database as fallback
    """
    codes = load_airline_codes()

    try:
        print("Downloading OpenFlights airlines.dat for extra mappings...")

        openflights = pd.read_csv(
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

        mask_real = (
            openflights["iata"].notna()
            & (openflights["iata"] != r"\N")
            & (openflights["iata"] != "")
        )

        openflights = openflights.loc[mask_real, ["name", "iata"]].copy()

        openflights["airline"] = (
            openflights["iata"]
            .astype("string")
            .str.upper()
            .str.strip()
        )

        openflights["airline_name_clean"] = (
            openflights["name"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        openflights["source_priority"] = 1

        combined = pd.concat(
            [
                codes[["airline", "airline_name_clean", "source_priority"]],
                openflights[["airline", "airline_name_clean", "source_priority"]],
            ],
            ignore_index=True,
        )

    except Exception as e:
        print("WARNING: Could not load OpenFlights. Using local airline codes only.")
        print("Reason:", e)

        combined = codes[["airline", "airline_name_clean", "source_priority"]]

    combined = (
        combined.dropna(subset=["airline_name_clean", "airline"])
        .sort_values("source_priority")
        .drop_duplicates(subset=["airline_name_clean"], keep="first")
    )

    mapping = combined.set_index("airline_name_clean")["airline"].to_dict()

    print(f"Built mapping for {len(mapping)} airline names.")

    return mapping


def main():
    # -----------------------------
    # Load master file
    # -----------------------------
    master_path = find_existing(MASTER_CANDIDATES)

    if master_path is None:
        raise FileNotFoundError(
            "Could not find any master file. Tried:\n"
            + "\n".join(str(p) for p in MASTER_CANDIDATES)
        )

    print(f"Loading master file: {master_path}")

    master = pd.read_csv(master_path, low_memory=False)

    print("Rows:", len(master))

    if "Airline" not in master.columns:
        raise ValueError("Master dataset is missing required column: Airline")

    # -----------------------------
    # Normalize stops
    # -----------------------------
    print("Filling numeric 'stops' from Stops_num / Stops...")

    if "Stops_num" in master.columns:
        master["stops"] = pd.to_numeric(master["Stops_num"], errors="coerce")
    else:
        master["stops"] = pd.NA

    if "Stops" in master.columns:
        mask_missing_stops = master["stops"].isna()

        if mask_missing_stops.any():
            parsed = master.loc[mask_missing_stops, "Stops"].map(parse_stops_text)
            master.loc[mask_missing_stops, "stops"] = parsed

    master["stops"] = pd.to_numeric(master["stops"], errors="coerce").astype("Int64")

    # -----------------------------
    # Airline mapping
    # -----------------------------
    name_to_code = build_name_to_code_mapping()

    print("Generating airline codes from Airline names...")

    if "airline" not in master.columns:
        master["airline"] = pd.NA

    master["Airline_clean"] = (
        master["Airline"]
        .astype("string")
        .map(clean_airline_name)
    )

    master["Airline_clean_lower"] = master["Airline_clean"].str.lower()

    master["airline_from_name"] = master["Airline_clean_lower"].map(name_to_code)

    master["airline"] = master["airline_from_name"].fillna(master["airline"])

    master["airline"] = (
        master["airline"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    master.loc[
        master["airline"].isin(["NAN", "NONE", ""]),
        "airline"
    ] = pd.NA

    # -----------------------------
    # Export unmatched airlines
    # -----------------------------
    print("Collecting unmatched airline rows / names...")

    mask_unmatched = master["airline"].isna() | (
        master["airline"].astype("string").str.strip() == ""
    )

    unmatched_rows = master.loc[mask_unmatched].copy()

    if not unmatched_rows.empty:
        UNMATCHED_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving {len(unmatched_rows)} unmatched rows to: {UNMATCHED_ROWS_PATH}")
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

        print(f"Saving {len(unique_names)} unmatched airline names to: {UNMATCHED_NAMES_PATH}")

        with open(UNMATCHED_NAMES_PATH, "w", encoding="utf-8") as f:
            for name in unique_names:
                f.write(name + "\n")

    else:
        print("No unmatched airlines remaining.")

    # -----------------------------
    # Save result
    # -----------------------------
    master = master.drop(
        columns=[
            "Airline_clean",
            "Airline_clean_lower",
            "airline_from_name",
        ],
        errors="ignore",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    master.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved final result to:")
    print(OUTPUT_PATH)
    print("Rows:", len(master))


if __name__ == "__main__":
    main()