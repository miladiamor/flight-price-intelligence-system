#!/usr/bin/env python3

import pandas as pd
from datetime import datetime, timedelta
import subprocess
import time
import sys
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # cheapflight-scraper-main/

ROUTES_FILE = BASE_DIR / "config" / "routes.csv"
SCRIPT = Path(__file__).resolve().parent / "scrape_google_flights.py"
CSV_OUTPUT = BASE_DIR / "data" / "raw" / "flights.csv"

# -----------------------------
# Settings
# -----------------------------
BATCH_SIZE = 40
DATE_RANGE_DAYS = 60


def _pick_url_column(df):
    for name in ("url", "urls", "link", "links"):
        if name in df.columns:
            return name
    return None


def main():
    try:
        df = pd.read_csv(ROUTES_FILE)
    except Exception as e:
        print(f"Could not read {ROUTES_FILE}: {e}")
        sys.exit(1)

    df.columns = [c.strip().lower() for c in df.columns]

    if not {"origin", "destination"}.issubset(df.columns):
        print("routes.csv must have at least columns: origin,destination (optional: url/urls/link)")
        sys.exit(1)

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    url_col = _pick_url_column(df)
    routes = df.to_dict("records")

    base = datetime.today()
    dates = [(base + timedelta(days=i)).date().isoformat() for i in range(DATE_RANGE_DAYS)]
    dates_arg = ",".join(dates)

    for i in range(0, len(routes), BATCH_SIZE):
        batch = routes[i:i + BATCH_SIZE]
        print(f"\n▶ Running batch {i // BATCH_SIZE + 1} ({len(batch)} routes)")

        for r in batch:
            origin = str(r.get("origin", "")).strip()
            dest = str(r.get("destination", "")).strip()

            sheet_url = str(r.get(url_col, "")).strip() if url_col else ""
            url = sheet_url if sheet_url else f"https://www.google.com/travel/flights?q={origin}+to+{dest}"

            cmd = [
                sys.executable,
                str(SCRIPT),
                "--url", url,
                "--dates", dates_arg,
                "--csv-output", str(CSV_OUTPUT),
                "--no-table",
                "--no-headless",
                "--first-screen-only",
                "--origin", origin,
                "--destination", dest,
            ]

            print(f"  ✈️  {origin} → {dest}")

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed: {origin} → {dest} ({e})")

            time.sleep(2)

    print("\n✅ All batches complete.")
    print(f"Saved output to: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()