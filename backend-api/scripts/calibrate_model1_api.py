from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _pick_col(df_cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(df_cols)
    for c in candidates:
        if c in s:
            return c
    return None


def infer_origin_dest_cols(df_cols: List[str]) -> Tuple[str, str]:
    origin_candidates = [
        "origin", "Origin", "ORIGIN",
        "origin_iata", "Origin_IATA", "ORIGIN_IATA",
        "IATA_origin", "iata_origin", "from", "From",
    ]
    dest_candidates = [
        "destination", "Destination", "DESTINATION",
        "dest", "Dest", "DEST",
        "destination_iata", "Destination_IATA", "DESTINATION_IATA",
        "IATA_destination", "iata_destination", "to", "To",
    ]
    ocol = _pick_col(df_cols, origin_candidates)
    dcol = _pick_col(df_cols, dest_candidates)
    if not ocol or not dcol:
        raise KeyError(
            f"Could not infer origin/destination columns.\n"
            f"Found columns: {df_cols[:80]}"
        )
    return ocol, dcol


def infer_date_col(df_cols: List[str]) -> Optional[str]:
    date_candidates = [
        "date", "Date", "travel_date", "TravelDate", "flight_date",
        "departure_date", "DepartureDate", "Date_flight",
        "date_flight", "flightDate"
    ]
    return _pick_col(df_cols, date_candidates)


def normalize_date_to_yyyy_mm_dd(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    try:
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def call_model1(api_url: str, origin: str, destination: str, date_str: str, timeout_s: int) -> Optional[float]:
    payload = {"origin": origin, "destination": destination, "date": date_str}
    try:
        r = requests.post(api_url, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            return None
        data = r.json()
        for key in ["predicted_price", "price", "prediction", "pred_price"]:
            if key in data:
                try:
                    return float(data[key])
                except Exception:
                    pass
        return None
    except Exception:
        return None

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = BASE_DIR / "data" / "model1_calibration.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to master_training_dataset.csv")
    ap.add_argument("--target", default="Price_num", help="Target column name (default Price_num)")
    ap.add_argument("--api", default="http://127.0.0.1:5001/model1", help="Model1 API endpoint")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output calibration JSON")
    ap.add_argument("--max_rows", type=int, default=200000, help="Read at most this many rows from CSV")
    ap.add_argument("--sample_n", type=int, default=3000, help="How many successful API predictions to collect")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--timeout", type=int, default=10, help="API timeout seconds")
    ap.add_argument("--sleep_ms", type=int, default=0, help="Optional sleep between calls (ms)")
    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset (up to {args.max_rows:,} rows): {data_path}")
    df = pd.read_csv(data_path, low_memory=False, nrows=args.max_rows)

    if args.target not in df.columns:
        raise KeyError(f"Target '{args.target}' not found. Columns: {list(df.columns)[:80]}")

    ocol, dcol = infer_origin_dest_cols(list(df.columns))
    date_col = infer_date_col(list(df.columns))

    if date_col is None:
        raise KeyError(
            "Could not find a travel date column in your CSV.\n"
            "Rename your date column to 'date' or one of: travel_date / flight_date / departure_date"
        )

    print(f"Using columns: origin={ocol}, destination={dcol}, date={date_col}, target={args.target}")
    df = df.dropna(subset=[args.target, ocol, dcol, date_col]).copy()

    df[ocol] = df[ocol].astype(str).str.strip()
    df[dcol] = df[dcol].astype(str).str.strip()
    df["__date_norm__"] = df[date_col].apply(normalize_date_to_yyyy_mm_dd)
    df = df.dropna(subset=["__date_norm__"]).copy()

    if df.empty:
        raise RuntimeError("After cleaning, no rows left to calibrate.")

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    y_true_list: List[float] = []
    y_pred_list: List[float] = []

    print(f"Collecting {args.sample_n} successful API predictions from: {args.api}")
    attempts = 0
    max_attempts = max(args.sample_n * 10, 20000)

    while len(y_pred_list) < args.sample_n and attempts < max_attempts and attempts < len(df):
        row = df.iloc[attempts]
        attempts += 1

        origin = row[ocol]
        destination = row[dcol]
        date_str = row["__date_norm__"]
        y_true = float(row[args.target])

        if len(origin) != 3 or len(destination) != 3:
            continue

        pred = call_model1(args.api, origin, destination, date_str, timeout_s=args.timeout)
        if pred is None or not np.isfinite(pred):
            continue

        y_true_list.append(y_true)
        y_pred_list.append(float(pred))

        if len(y_pred_list) % 200 == 0:
            print(f"  collected={len(y_pred_list):,} / {args.sample_n:,} (attempts={attempts:,})")

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    if len(y_pred_list) < 500:
        raise RuntimeError(
            f"Too few successful API predictions: {len(y_pred_list)}.\n"
            f"Check backend is running and your CSV routes/dates match the model."
        )

    y_pred = np.array(y_pred_list, dtype=float)
    y_true = np.array(y_true_list, dtype=float)

    print(f"\nFitting calibration on sample_size={len(y_pred):,} ...")
    reg = LinearRegression(fit_intercept=True)
    reg.fit(y_pred.reshape(-1, 1), y_true)

    a = float(reg.coef_[0])
    b = float(reg.intercept_)
    y_cal = a * y_pred + b

    mae_raw = float(mean_absolute_error(y_true, y_pred))
    mae_cal = float(mean_absolute_error(y_true, y_cal))

    # ✅ compatible RMSE (no 'squared=' needed)
    rmse_raw = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmse_cal = float(np.sqrt(mean_squared_error(y_true, y_cal)))

    r2_raw = float(r2_score(y_true, y_pred))
    r2_cal = float(r2_score(y_true, y_cal))

    print("\n=== Calibration learned ===")
    print(f"a (scale)  = {a:.6f}")
    print(f"b (offset) = {b:.6f}")

    print("\n=== Sample metrics ===")
    print(f"MAE  raw   = {mae_raw:.3f}")
    print(f"MAE  calib = {mae_cal:.3f}")
    print(f"RMSE raw   = {rmse_raw:.3f}")
    print(f"RMSE calib = {rmse_cal:.3f}")
    print(f"R²   raw   = {r2_raw:.4f}")
    print(f"R²   calib = {r2_cal:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "api_calibration",
        "api": args.api,
        "target_col": args.target,
        "a": a,
        "b": b,
        "sample_size": int(len(y_pred)),
        "metrics_sample": {
            "mae_raw": mae_raw,
            "mae_calibrated": mae_cal,
            "rmse_raw": rmse_raw,
            "rmse_calibrated": rmse_cal,
            "r2_raw": r2_raw,
            "r2_calibrated": r2_cal,
        },
        "notes": "Calibration fitted as y_true ~= a*y_pred + b using /model1 predictions."
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n✅ Saved calibration to: {out_path}")


if __name__ == "__main__":
    main()
