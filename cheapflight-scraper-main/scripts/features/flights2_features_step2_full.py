#!/usr/bin/env python3
"""
Feature Step 2: Add engineered features for price modeling.

Reads:
    data/final/flights_features_step1_daysleft.csv

Adds:
    Day_of_week
    Month_of_year
    Is_weekend
    Departure_hour
    Duration_minutes
    Stops_num
    Route_popularity
    Origin_country
    Destination_country
    Origin_continent
    Destination_continent
    Travel_direction
    Airline_encoded
    Price_per_mile
    Days_into_month
    High_season_flag

Writes:
    data/final/flights_features_step2_full.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # cheapflight-scraper-main/

INPUT_FILE = BASE_DIR / "data" / "final" / "flights_features_step1_daysleft.csv"
OUTPUT_FILE = BASE_DIR / "data" / "final" / "flights_features_step2_full.csv"


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


def find_departure_column(df: pd.DataFrame) -> str:
    candidates = [
        "Departure Time",
        "Departure Ti",
        "Departure",
        "Dep_Time",
        "dep_time",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        if "depart" in col.lower():
            return col

    raise ValueError("Could not find a departure time column.")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print("Loaded:", INPUT_FILE)
    print("Rows:", len(df))

    required_cols = ["Date", "Search Date", "Origin", "Destination", "Airline", "Price_num"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -----------------------------
    # Date features
    # -----------------------------
    flight_dt = robust_parse(df["Date"], "Date")
    search_dt = robust_parse(df["Search Date"], "Search Date")

    df["Day_of_week"] = flight_dt.dt.weekday
    df["Month_of_year"] = flight_dt.dt.month
    df["Days_into_month"] = flight_dt.dt.day
    df["Is_weekend"] = df["Day_of_week"].isin([5, 6]).astype(int)
    df["High_season_flag"] = df["Month_of_year"].isin([6, 7, 8, 12]).astype(int)

    if "Days_left" not in df.columns:
        df["Days_left"] = (flight_dt - search_dt).dt.days

    # -----------------------------
    # Departure hour
    # -----------------------------
    dep_col = find_departure_column(df)
    print(f"Using departure time column: {dep_col}")

    dep_times = pd.to_datetime(
        df[dep_col].astype("string").str.strip(),
        errors="coerce"
    )

    df["Departure_hour"] = dep_times.dt.hour

    # -----------------------------
    # Duration
    # -----------------------------
    if "Duration_min" in df.columns:
        df["Duration_minutes"] = pd.to_numeric(df["Duration_min"], errors="coerce")
    elif "Duration_minutes" in df.columns:
        df["Duration_minutes"] = pd.to_numeric(df["Duration_minutes"], errors="coerce")
    elif "Duration" in df.columns:
        dur_str = df["Duration"].astype("string")
        hours = dur_str.str.extract(r"(\d+)\s*hr", expand=False).astype(float)
        mins = dur_str.str.extract(r"(\d+)\s*min", expand=False).astype(float)
        df["Duration_minutes"] = hours.fillna(0) * 60 + mins.fillna(0)
    else:
        raise ValueError("Could not find duration column.")

    # -----------------------------
    # Stops
    # -----------------------------
    if "Stops_num" in df.columns:
        df["Stops_num"] = pd.to_numeric(df["Stops_num"], errors="coerce").fillna(0).astype(int)
    elif "Stops" in df.columns:
        stops_str = df["Stops"].astype("string").str.lower().str.strip()
        df["Stops_num"] = 0
        df.loc[stops_str.str.contains("1 stop", na=False), "Stops_num"] = 1
        df.loc[stops_str.str.contains("2 stop", na=False), "Stops_num"] = 2
        df.loc[stops_str.str.contains("3 stop", na=False), "Stops_num"] = 3
    else:
        raise ValueError("Could not find Stops_num or Stops column.")

    # -----------------------------
    # Route popularity
    # -----------------------------
    if "Route_popularity" in df.columns:
        df = df.drop(columns=["Route_popularity"])

    route_counts = df.groupby(["Origin", "Destination"]).size().rename("Route_popularity")
    df = df.join(route_counts, on=["Origin", "Destination"])

    # -----------------------------
    # Simple country / continent mapping
    # Note: this is basic fallback mapping.
    # Full route enrichment happens later in enrichment scripts.
    # -----------------------------
    airport_to_country = {
        "JFK": "USA",
        "PHX": "USA",
        "DOH": "Qatar",
        "OSL": "Norway",
        "BER": "Germany",
        "FRA": "Germany",
        "MUC": "Germany",
        "HAM": "Germany",
        "CDG": "France",
        "LHR": "United Kingdom",
        "AMS": "Netherlands",
        "BCN": "Spain",
        "MAD": "Spain",
    }

    country_to_continent = {
        "USA": "North America",
        "Qatar": "Asia",
        "Norway": "Europe",
        "Germany": "Europe",
        "France": "Europe",
        "United Kingdom": "Europe",
        "Netherlands": "Europe",
        "Spain": "Europe",
    }

    df["Origin_country"] = df["Origin"].map(airport_to_country).fillna("Unknown")
    df["Destination_country"] = df["Destination"].map(airport_to_country).fillna("Unknown")

    df["Origin_continent"] = df["Origin_country"].map(country_to_continent).fillna("Unknown")
    df["Destination_continent"] = df["Destination_country"].map(country_to_continent).fillna("Unknown")

    df["Travel_direction"] = (
        df["Origin_country"].astype(str) != df["Destination_country"].astype(str)
    ).astype(int)

    # -----------------------------
    # Airline encoding
    # -----------------------------
    df["Airline_encoded"] = df["Airline"].astype("category").cat.codes

    # -----------------------------
    # Price per duration
    # -----------------------------
    df["Price_num"] = pd.to_numeric(df["Price_num"], errors="coerce")
    df["Duration_minutes"] = pd.to_numeric(df["Duration_minutes"], errors="coerce")

    df["Price_per_mile"] = df["Price_num"] / df["Duration_minutes"]
    df["Price_per_mile"] = df["Price_per_mile"].replace([np.inf, -np.inf], np.nan)

    if df["Price_per_mile"].notna().any():
        df["Price_per_mile"] = df["Price_per_mile"].fillna(df["Price_per_mile"].median())
    else:
        df["Price_per_mile"] = 0.0

    # -----------------------------
    # Save
    # -----------------------------
    feature_cols = [
        "Days_left",
        "Day_of_week",
        "Month_of_year",
        "Is_weekend",
        "Departure_hour",
        "Duration_minutes",
        "Stops_num",
        "Route_popularity",
        "Origin_country",
        "Destination_country",
        "Origin_continent",
        "Destination_continent",
        "Travel_direction",
        "Airline_encoded",
        "Price_per_mile",
        "Days_into_month",
        "High_season_flag",
    ]

    print("\n=== Feature summary ===")
    print("Created features:")
    for col in feature_cols:
        print("-", col)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved file ->", OUTPUT_FILE)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()