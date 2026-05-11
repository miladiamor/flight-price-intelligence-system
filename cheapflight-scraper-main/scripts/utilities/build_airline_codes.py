import pandas as pd
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# paths – adjust if needed
MASTER_PATH        = os.path.join(BASE_DIR, "master_training_dataset_enriched_v2_airline_stops_filled.csv")
EXISTING_MAP_PATH  = os.path.join(BASE_DIR, "airline_codes.csv")
OUTPUT_MAP_PATH    = os.path.join(BASE_DIR, "airline_codes_extended.csv")
UNMATCHED_LIST_PATH= os.path.join(BASE_DIR, "airline_codes_unmatched.txt")

# openflights airlines dataset URL
OPENFLIGHTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"

print("Loading master dataset …")
master = pd.read_csv(MASTER_PATH, dtype={"airline": str})
master["airline"] = master["airline"].str.upper().fillna("")

print("Loading existing airline codes mapping …")
existing = pd.read_csv(EXISTING_MAP_PATH, dtype=str)
existing["airline"] = existing["airline"].str.upper()

print("Downloading OpenFlights dataset …")
of = pd.read_csv(OPENFLIGHTS_URL, header=None, names=[
    "AirlineID", "Name", "Alias", "IATA", "ICAO", "Callsign", "Country", "Active"
])

# keep only rows with valid IATA/2-letter code
of2 = of[(of["IATA"].notna()) & (of["IATA"] != "\\N") & (of["IATA"] != "")]
of2 = of2.copy()
of2["airline"] = of2["IATA"].str.upper()
of2["airline_name"] = of2["Name"]
of2 = of2[["airline", "airline_name"]].drop_duplicates().reset_index(drop=True)

print(f"OpenFlights mapping has {len(of2)} codes")

print("Merging with existing mapping …")
merged_map = existing.merge(of2, on="airline", how="outer", suffixes=("_orig", "_of"))

# If airline_name_orig is missing, take airline_name_of
merged_map["airline_name"] = merged_map["airline_name_orig"].fillna(merged_map["airline_name_of"])

# For unmapped codes, airline_name may still be NaN
merged_map = merged_map[["airline", "airline_name"]].sort_values("airline").reset_index(drop=True)

print(f"Extended mapping has {len(merged_map)} codes; saving to {OUTPUT_MAP_PATH}")
merged_map.to_csv(OUTPUT_MAP_PATH, index=False)

print("Finding unmatched codes in master …")
codes_in_master = set(master["airline"].unique())
codes_in_map    = set(merged_map["airline"].dropna().unique())
unmatched = sorted(codes_in_master - codes_in_map)

print(f"{len(unmatched)} unmatched airline codes found. Saving to {UNMATCHED_LIST_PATH}")
with open(UNMATCHED_LIST_PATH, "w") as f:
    for code in unmatched:
        if code and str(code).strip():
            f.write(f"{code}\n")

print("Done.")
