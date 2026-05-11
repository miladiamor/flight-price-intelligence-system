import pandas as pd
import os
from pathlib import Path
import re

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MASTER_PATH = os.path.join(
    BASE_DIR, "master_training_dataset_enriched_v2_fixed.csv"
)
AIRLINE_CODES_PATH = os.path.join(
    BASE_DIR, "airline_codes.csv"
)
OUTPUT_PATH = os.path.join(
    BASE_DIR, "master_training_dataset_enriched_v2_airline_stops_filled.csv"
)

# files for option 3 (manual fixing)
UNMATCHED_ROWS_PATH = os.path.join(
    BASE_DIR, "unmatched_airlines_rows.csv"
)
UNMATCHED_NAMES_PATH = os.path.join(
    BASE_DIR, "unmatched_airline_names.txt"
)

# OpenFlights airlines database (option 2)
OPENFLIGHTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"
# -----------------------------


def clean_airline_name(s: str) -> str:
    """
    Take the scraped Airline text and simplify it so it matches airline_codes/openflights.
    Examples:
      'KLM' -> 'KLM'
      'Qatar Airways + JAL' -> 'Qatar Airways'
      'Scandinavian Airlines + Operated by Sas Connect' -> 'Scandinavian Airlines'
      'Air France, WestJet + Operated by HOP!' -> 'Air France'
    """
    if pd.isna(s):
        return pd.NA
    s = str(s)

    # cut off after 'operated by'
    s = re.split(r'\boperated by\b', s, flags=re.IGNORECASE)[0]

    # cut at first comma / plus / slash
    s = re.split(r'[+,/]', s)[0]

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


def build_name_to_code_mapping() -> dict:
    """
    Build airline_name_clean -> IATA code mapping from:
      1) airline_codes.csv (user file, takes priority)
      2) OpenFlights airlines database (fallback)
    """
    print("Loading airline_codes.csv...")
    codes = pd.read_csv(AIRLINE_CODES_PATH, dtype="string")

    if "airline" not in codes.columns or "airline_name" not in codes.columns:
        raise KeyError("airline_codes.csv must contain 'airline' and 'airline_name' columns.")

    codes["airline"] = codes["airline"].astype("string").str.upper().str.strip()
    codes["airline_name_clean"] = (
        codes["airline_name"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    codes["source_priority"] = 0  # user file highest priority

    # Try to enrich with OpenFlights (option 2)
    try:
        print("Downloading OpenFlights airlines.dat for extra mappings (option 2)...")
        of = pd.read_csv(
            OPENFLIGHTS_URL,
            header=None,
            names=["id", "name", "alias", "iata", "icao", "callsign", "country", "active"],
        )

        # only rows with real 2-letter IATA code
        mask_real = of["iata"].notna() & (of["iata"] != r"\N") & (of["iata"] != "")
        of = of.loc[mask_real, ["name", "iata"]].copy()

        of["airline"] = of["iata"].astype("string").str.upper().str.strip()
        of["airline_name_clean"] = (
            of["name"].astype("string").str.strip().str.lower()
        )
        of["source_priority"] = 1  # lower priority than user file

        combined = pd.concat(
            [
                codes[["airline", "airline_name_clean", "source_priority"]],
                of[["airline", "airline_name_clean", "source_priority"]],
            ],
            ignore_index=True,
        )

        # sort so user-file rows come first, then drop duplicates on name
        combined = (
            combined.sort_values("source_priority")
            .drop_duplicates(subset=["airline_name_clean"], keep="first")
        )

        mapping = (
            combined.dropna(subset=["airline_name_clean", "airline"])
            .set_index("airline_name_clean")["airline"]
            .to_dict()
        )
        print(f"Built mapping for {len(mapping)} airline names (user + OpenFlights).")
        return mapping

    except Exception as e:
        print("WARNING: Could not load OpenFlights airlines.dat; using airline_codes.csv only.")
        print("Reason:", e)
        mapping = (
            codes.dropna(subset=["airline_name_clean", "airline"])
            .drop_duplicates(subset=["airline_name_clean"])
            .set_index("airline_name_clean")["airline"]
            .to_dict()
        )
        print(f"Built mapping for {len(mapping)} airline names (user file only).")
        return mapping


def main():
    print("Loading master file...")
    master = pd.read_csv(MASTER_PATH)

    # ------- STOP VALUES -------
    print("Filling numeric 'stops' from Stops_num / Stops...")
    if "Stops_num" in master.columns:
        master["stops"] = master["Stops_num"]
    else:
        master["stops"] = pd.NA

    if "Stops" in master.columns:
        mask_missing_stops = master["stops"].isna()
        if mask_missing_stops.any():
            parsed = master.loc[mask_missing_stops, "Stops"].map(parse_stops_text)
            master.loc[mask_missing_stops, "stops"] = parsed

    master["stops"] = master["stops"].astype("Int64")

    # ------- AIRLINE CODES (pass 1 + option 2) -------
    name_to_code = build_name_to_code_mapping()

    print("Generating airline codes from Airline names...")
    if "airline" not in master.columns:
        master["airline"] = pd.NA

    master["Airline_clean"] = master["Airline"].astype("string").map(clean_airline_name)
    master["Airline_clean_lower"] = master["Airline_clean"].str.lower()

    master["airline_from_name"] = master["Airline_clean_lower"].map(name_to_code)

    master["airline"] = master["airline_from_name"].fillna(master["airline"])
    master["airline"] = master["airline"].astype("string").str.upper().str.strip()
    master.loc[master["airline"] == "nan", "airline"] = pd.NA

    # ------- OPTION 3: EXPORT UNMATCHED AIRLINES -------
    print("Collecting unmatched airline rows / names (option 3)...")
    mask_unmatched = master["airline"].isna() | (master["airline"].astype("string").str.strip() == "")
    unmatched_rows = master.loc[mask_unmatched].copy()

    if not unmatched_rows.empty:
        print(f"Saving {len(unmatched_rows)} unmatched rows to {UNMATCHED_ROWS_PATH}")
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

        print(f"Saving {len(unique_names)} unmatched airline names to {UNMATCHED_NAMES_PATH}")
        with open(UNMATCHED_NAMES_PATH, "w", encoding="utf-8") as f:
            for name in unique_names:
                f.write(name + "\n")
    else:
        print("No unmatched airlines remaining – nice!")

    # drop helper columns before saving main file
    master = master.drop(columns=["Airline_clean", "Airline_clean_lower", "airline_from_name"], errors="ignore")

    print(f"Saving result to: {OUTPUT_PATH}")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT_PATH, index=False)
    print("Done! (If you get PermissionError, close the CSV in Excel and run again.)")


if __name__ == "__main__":
    main()
