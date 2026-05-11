# services/model1_service.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
import json


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # backend_big/
MODELS_DIR = BASE_DIR / "models"

MODEL1_PATH = MODELS_DIR / "model1_pipeline_fast.pkl"
META1_PATH = MODELS_DIR / "model1_metadata_fast.pkl"

AIRPORTS_PATH = BASE_DIR / "data" / "airports.csv"
ROUTES_PATH = BASE_DIR / "data" / "routes_features_final_continents_filled.csv"

# Calibration output created by scripts/calibrate_model1_api.py
CALIB_PATH = BASE_DIR / "data" / "model1_calibration.json"


# -----------------------------
# Load model artifacts once
# -----------------------------
@dataclass
class Model1Artifacts:
    pipeline: Any
    meta: Dict[str, Any]


def _load_model1() -> Model1Artifacts:
    if not MODEL1_PATH.exists():
        raise FileNotFoundError(f"Missing file: {MODEL1_PATH}")
    if not META1_PATH.exists():
        raise FileNotFoundError(f"Missing file: {META1_PATH}")

    pipeline = joblib.load(MODEL1_PATH)
    meta = joblib.load(META1_PATH)
    return Model1Artifacts(pipeline=pipeline, meta=meta)


ART = _load_model1()


# -----------------------------
# Calibration cache
# price_calib = a * price_raw + b
# -----------------------------
_CALIB: Optional[Dict[str, float]] = None


def _load_calibration() -> Optional[Dict[str, float]]:
    global _CALIB
    if _CALIB is not None:
        return _CALIB

    if not CALIB_PATH.exists():
        _CALIB = None
        return None

    try:
        obj = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        a = float(obj.get("a", 1.0))
        b = float(obj.get("b", 0.0))
        _CALIB = {"a": a, "b": b}
        return _CALIB
    except Exception:
        _CALIB = None
        return None


def _apply_calibration(price_raw: float) -> float:
    cal = _load_calibration()
    if not cal:
        return float(price_raw)
    return float(cal["a"] * float(price_raw) + cal["b"])


# -----------------------------
# Airports cache
# airports.csv expected columns:
# iata, city, airport_name, country, continent, lat, lon
# -----------------------------
_AIRPORTS: Optional[pd.DataFrame] = None


def _load_airports() -> pd.DataFrame:
    global _AIRPORTS
    if _AIRPORTS is not None:
        return _AIRPORTS

    if not AIRPORTS_PATH.exists():
        raise FileNotFoundError(
            f"airports.csv not found at: {AIRPORTS_PATH}\n"
            f"Place airports.csv there or update AIRPORTS_PATH in model1_service.py"
        )

    df = pd.read_csv(AIRPORTS_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    needed = {"iata", "city", "airport_name", "country", "continent", "lat", "lon"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"airports.csv missing columns: {sorted(list(missing))}")

    df["iata"] = df["iata"].astype(str).str.strip().str.upper()

    for c in ["country", "continent", "city", "airport_name"]:
        df[c] = df[c].astype(str).str.strip()

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    _AIRPORTS = df
    return _AIRPORTS


def _airport_row(iata: str) -> Optional[pd.Series]:
    airports = _load_airports()
    code = str(iata).strip().upper()
    hit = airports.loc[airports["iata"] == code]
    if hit.empty:
        return None
    return hit.iloc[0]


# -----------------------------
# Routes cache
# -----------------------------
_ROUTES: Optional[pd.DataFrame] = None
_ROUTE_KEY_ORIG: Optional[str] = None
_ROUTE_KEY_DEST: Optional[str] = None


def _pick_first_existing(cols: list[str], candidates: list[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _load_routes() -> pd.DataFrame:
    global _ROUTES, _ROUTE_KEY_ORIG, _ROUTE_KEY_DEST
    if _ROUTES is not None:
        return _ROUTES

    if not ROUTES_PATH.exists():
        raise FileNotFoundError(
            f"routes_features_final_continents_filled.csv not found at: {ROUTES_PATH}\n"
            f"Place the file there or update ROUTES_PATH in model1_service.py"
        )

    df = pd.read_csv(ROUTES_PATH, low_memory=False)
    cols = list(df.columns)

    orig_col = _pick_first_existing(
        cols,
        ["origin_iata", "Origin", "origin", "origin_iata_code", "origin_iata_co", "Origin_iata", "origin_iata_code"],
    )
    dest_col = _pick_first_existing(
        cols,
        ["destination_iata", "Destination", "destination", "dest_iata_code", "dest_iata_co", "Destination_iata"],
    )

    if not orig_col or not dest_col:
        raise ValueError(
            "Could not find origin/destination columns in routes_features file.\n"
            f"Columns I saw: {cols[:40]}...\n"
            "Make sure it has something like origin_iata and destination_iata (or similar)."
        )

    df[orig_col] = df[orig_col].astype(str).str.strip().str.upper()
    df[dest_col] = df[dest_col].astype(str).str.strip().str.upper()

    df["_route_key"] = df[orig_col] + "_" + df[dest_col]
    df = df.drop_duplicates(subset=["_route_key"]).set_index("_route_key", drop=False)

    _ROUTES = df
    _ROUTE_KEY_ORIG = orig_col
    _ROUTE_KEY_DEST = dest_col
    return _ROUTES


def _route_row(origin: str, destination: str) -> Optional[pd.Series]:
    df = _load_routes()
    key = f"{origin}_{destination}"
    if key not in df.index:
        return None
    return df.loc[key]


# -----------------------------
# Utilities
# -----------------------------
def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(R * c)


def _distance_class(distance_km: float) -> str:
    if distance_km < 1500:
        return "short"
    if distance_km < 3500:
        return "medium"
    return "long"


def _travel_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    ns = "N" if lat2 >= lat1 else "S"
    ew = "E" if lon2 >= lon1 else "W"
    return ns + ew


def _is_high_season(month: int) -> int:
    return 1 if month in (6, 7, 8, 12) else 0


def _is_weekend(d: date) -> int:
    return 1 if d.weekday() >= 5 else 0


def _safe_get(r: pd.Series, names: list[str]) -> Any:
    for n in names:
        if n in r.index:
            return r[n]
    low = {str(k).lower(): k for k in r.index}
    for n in names:
        k = low.get(n.lower())
        if k is not None:
            return r[k]
    return None


# -----------------------------
# Build feature row
# -----------------------------
def _build_feature_row(
    origin: str,
    destination: str,
    date_str: str,
    *,
    search_date_override: date | None = None,
) -> Tuple[pd.DataFrame, int]:
    origin = str(origin).strip().upper()
    destination = str(destination).strip().upper()
    travel_date = _parse_date(date_str)

    today = search_date_override or date.today()
    dleft_signed = (travel_date - today).days

    rr = _route_row(origin, destination)
    o = _airport_row(origin)
    d = _airport_row(destination)

    distance_km = None
    if rr is not None:
        distance_km = _safe_get(rr, ["distance_km", "Distance_km", "dist_km"])
    distance_km = pd.to_numeric(distance_km, errors="coerce")

    o_lat = None
    o_lon = None
    d_lat = None
    d_lon = None

    if rr is not None:
        o_lat = _safe_get(rr, ["origin_latitude_deg", "origin_lat", "orig_lat"])
        o_lon = _safe_get(rr, ["origin_longitude_deg", "origin_lon", "orig_lon"])
        d_lat = _safe_get(rr, ["dest_latitude_deg", "dest_lat", "destination_lat"])
        d_lon = _safe_get(rr, ["dest_longitude_deg", "dest_lon", "destination_lon"])

    if o_lat is None and o is not None:
        o_lat = o["lat"]
    if o_lon is None and o is not None:
        o_lon = o["lon"]
    if d_lat is None and d is not None:
        d_lat = d["lat"]
    if d_lon is None and d is not None:
        d_lon = d["lon"]

    o_lat = pd.to_numeric(o_lat, errors="coerce")
    o_lon = pd.to_numeric(o_lon, errors="coerce")
    d_lat = pd.to_numeric(d_lat, errors="coerce")
    d_lon = pd.to_numeric(d_lon, errors="coerce")

    if (pd.isna(distance_km) or distance_km is None) and np.isfinite([o_lat, o_lon, d_lat, d_lon]).all():
        distance_km = _haversine_km(float(o_lat), float(o_lon), float(d_lat), float(d_lon))

    if pd.isna(distance_km):
        distance_km = np.nan

    dist_class = "Unknown"
    if rr is not None:
        dc = _safe_get(rr, ["distance_class", "Distance_class"])
        if dc is not None and str(dc).strip() != "":
            dist_class = str(dc).strip().lower()
    if dist_class == "Unknown" and np.isfinite(distance_km):
        dist_class = _distance_class(float(distance_km))

    origin_country = "Unknown"
    dest_country = "Unknown"
    origin_continent = "Unknown"
    dest_continent = "Unknown"
    continent_pair = "Unknown"

    if rr is not None:
        origin_country = _safe_get(rr, ["origin_iso_country", "Origin_country", "origin_country"]) or origin_country
        dest_country = _safe_get(rr, ["dest_iso_country", "Destination_country", "destination_country"]) or dest_country
        origin_continent = _safe_get(rr, ["origin_continent", "Origin_continent"]) or origin_continent
        dest_continent = _safe_get(rr, ["dest_continent", "Destination_continent"]) or dest_continent
        continent_pair = _safe_get(rr, ["continent_pair"]) or continent_pair

    if origin_country == "Unknown" and o is not None:
        origin_country = str(o["country"])
    if dest_country == "Unknown" and d is not None:
        dest_country = str(d["country"])
    if origin_continent == "Unknown" and o is not None:
        origin_continent = str(o["continent"])
    if dest_continent == "Unknown" and d is not None:
        dest_continent = str(d["continent"])

    if continent_pair == "Unknown" and origin_continent != "Unknown" and dest_continent != "Unknown":
        continent_pair = f"{origin_continent}_{dest_continent}"

    domestic = 1 if (origin_country != "Unknown" and origin_country == dest_country) else 0

    direction_lat = np.nan
    direction_lon = np.nan
    travel_direction = "Unknown"

    if rr is not None:
        direction_lat = _safe_get(rr, ["direction_lat"]) or direction_lat
        direction_lon = _safe_get(rr, ["direction_lon"]) or direction_lon
        travel_direction = _safe_get(rr, ["Travel_direction", "travel_direction", "travel_dir"]) or travel_direction

    if travel_direction == "Unknown" and np.isfinite([o_lat, o_lon, d_lat, d_lon]).all():
        travel_direction = _travel_direction(float(o_lat), float(o_lon), float(d_lat), float(d_lon))
        direction_lat = float(d_lat - o_lat)
        direction_lon = float(d_lon - o_lon)

    duration_minutes = None
    stops_num = None

    if rr is not None:
        duration_minutes = _safe_get(rr, ["Duration_minutes", "duration_minutes", "Duration_min", "Duration"])
        stops_num = _safe_get(rr, ["Stops_num", "stops", "Stops"])

    duration_minutes = pd.to_numeric(duration_minutes, errors="coerce")
    stops_num = pd.to_numeric(stops_num, errors="coerce")

    if pd.isna(duration_minutes) and np.isfinite(distance_km):
        duration_minutes = max(40.0, (float(distance_km) / 800.0) * 60.0 + 60.0)

    if pd.isna(stops_num):
        if np.isfinite(distance_km):
            stops_num = 0 if float(distance_km) < 4500 else 1
        else:
            stops_num = 1

    stops_num = int(stops_num)
    is_nonstop = 1 if stops_num == 0 else 0
    is_onestop = 1 if stops_num == 1 else 0
    is_multistop = 1 if stops_num >= 2 else 0

    duration_min = float(duration_minutes) if pd.notna(duration_minutes) else np.nan
    duration_mi = (float(distance_km) * 0.621371) if np.isfinite(distance_km) else np.nan

    origin_hub_score = 0.0
    dest_hub_score = 0.0
    origin_connectivity = 0.0
    dest_connectivity = 0.0
    competition_score = 0.0
    route_popularity = 0.0
    longhaul_capable = 1 if dist_class == "long" else 0

    if rr is not None:
        origin_hub_score = float(pd.to_numeric(_safe_get(rr, ["origin_hub_score"]), errors="coerce") or 0.0)
        dest_hub_score = float(pd.to_numeric(_safe_get(rr, ["dest_hub_score"]), errors="coerce") or 0.0)
        origin_connectivity = float(pd.to_numeric(_safe_get(rr, ["origin_connectivity"]), errors="coerce") or 0.0)
        dest_connectivity = float(pd.to_numeric(_safe_get(rr, ["dest_connectivity"]), errors="coerce") or 0.0)
        competition_score = float(pd.to_numeric(_safe_get(rr, ["competition_score"]), errors="coerce") or 0.0)

        rp = _safe_get(rr, ["route_popularity", "Route_popularity", "route_popular", "route_popularity_score"])
        route_popularity = float(pd.to_numeric(rp, errors="coerce") or 0.0)

        lh = _safe_get(rr, ["longhaul_capable"])
        if lh is not None and str(lh).strip() != "":
            longhaul_capable = int(pd.to_numeric(lh, errors="coerce") or longhaul_capable)

    month = travel_date.month
    day_of_week = travel_date.weekday()
    days_into_month = travel_date.day
    high_season_flag = _is_high_season(month)

    date_search = today.strftime("%Y-%m-%d")

    origin_name = str(o["airport_name"]) if o is not None else "Unknown"
    dest_name = str(d["airport_name"]) if d is not None else "Unknown"
    origin_city = str(o["city"]) if o is not None else "Unknown"
    dest_city = str(d["city"]) if d is not None else "Unknown"

    route = f"{origin}_{destination}"

    row = {
        "Origin": origin,
        "Destination": destination,
        "origin_iata": origin,
        "destination_iata": destination,
        "origin_iata_code": origin,
        "dest_iata_code": destination,

        "Date": travel_date.strftime("%Y-%m-%d"),
        "date_search": date_search,

        "Departure Time": "12:00",
        "Arrival Time": "18:00",
        "Departure_hour": 12,

        "Days_left": max(dleft_signed, 0),
        "Day_of_week": day_of_week,
        "Month_of_year": month,
        "Days_into_month": days_into_month,
        "Is_weekend": _is_weekend(travel_date),
        "High_season_flag": high_season_flag,

        "origin_latitude_deg": float(o_lat) if pd.notna(o_lat) else np.nan,
        "origin_longitude_deg": float(o_lon) if pd.notna(o_lon) else np.nan,
        "dest_latitude_deg": float(d_lat) if pd.notna(d_lat) else np.nan,
        "dest_longitude_deg": float(d_lon) if pd.notna(d_lon) else np.nan,

        "distance_km": float(distance_km) if np.isfinite(distance_km) else np.nan,
        "distance_class": dist_class,
        "domestic": domestic,
        "Origin_country": str(origin_country),
        "Destination_country": str(dest_country),
        "Origin_continent": str(origin_continent),
        "Destination_continent": str(dest_continent),
        "continent_pair": str(continent_pair),

        "direction_lat": float(direction_lat) if pd.notna(direction_lat) else np.nan,
        "direction_lon": float(direction_lon) if pd.notna(direction_lon) else np.nan,
        "Travel_direction": str(travel_direction),

        "Duration_minutes": float(duration_minutes) if pd.notna(duration_minutes) else np.nan,
        "Duration_min": float(duration_min) if pd.notna(duration_min) else np.nan,
        "Duration_mi": float(duration_mi) if pd.notna(duration_mi) else np.nan,

        "Stops_num": int(stops_num),
        "Stops": str(int(stops_num)),
        "stops": int(stops_num),
        "Is_nonstop": is_nonstop,
        "Is_onestop": is_onestop,
        "Is_multistop": is_multistop,

        "origin_name": origin_name,
        "dest_name": dest_name,
        "origin_municipality": origin_city,
        "dest_municipality": dest_city,
        "origin_iso_country": str(origin_country),
        "dest_iso_country": str(dest_country),
        "origin_continent": str(origin_continent),
        "dest_continent": str(dest_continent),
        "dest_type": "Unknown",
        "origin_type": "Unknown",

        "Airline": "Unknown",
        "airline": "Unknown",
        "airline_type": "Unknown",
        "airline_region": "Unknown",
        "aircraft_class": "Unknown",
        "equipment": "Unknown",

        "Route_popularity": float(route_popularity),
        "route_popularity": float(route_popularity),
        "competition_score": float(competition_score),
        "origin_hub_score": float(origin_hub_score),
        "dest_hub_score": float(dest_hub_score),
        "origin_connectivity": float(origin_connectivity),
        "dest_connectivity": float(dest_connectivity),
        "airline_route_count": 0.0,
        "airline_route_share": 0.0,
        "longhaul_capable": int(longhaul_capable),

        "route": route,
    }

    return pd.DataFrame([row]), int(dleft_signed)


# -----------------------------
# Prepare input for pipeline based on meta
# -----------------------------
def _prepare_for_model1(df_raw: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    feature_cols = meta["feature_cols"]
    numeric_cols = meta["numeric_cols"]
    categorical_cols = meta["categorical_cols"]

    df = df_raw.copy()

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].replace(["nan", "None", "NaT"], "Unknown")
        df[col] = df[col].fillna("Unknown")

    return df[feature_cols]


# -----------------------------
# Flat-curve fix (smoothing overlay)
# -----------------------------
def _is_curve_flat(prices: List[float]) -> bool:
    """Detect 'piecewise constant' curves in next-7-days output."""
    if len(prices) <= 2:
        return True
    arr = np.array(prices, dtype=float)
    mean = float(np.mean(arr))
    if mean <= 0:
        return True
    std = float(np.std(arr))
    # Flat if std is tiny in absolute and relative terms
    return (std < 0.25) or (std / mean < 0.0025)  # 0.25 EUR or <0.25% variation


def _smoothing_rate(dist_class: str, domestic: int) -> float:
    """
    Deterministic tiny drift per wait-day.
    Aim: avoid perfectly flat curves while staying realistic.
    """
    dc = (dist_class or "").lower().strip()
    if dc not in {"short", "medium", "long"}:
        dc = "short"

    # Base daily drift (fraction of price)
    base = {
        "short": 0.0015,   # 0.15% / day
        "medium": 0.0010,  # 0.10% / day
        "long": 0.0007,    # 0.07% / day
    }[dc]

    # Domestic tends to be a bit more stable
    if int(domestic) == 1:
        base *= 0.85

    return float(base)


def _apply_curve_smoothing(
    points: List[Dict[str, Any]],
    *,
    dist_class: str,
    domestic: int,
) -> Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
    """
    If the calibrated curve is essentially flat, apply a gentle monotonic drift:
      adjusted_price = price * (1 + rate * wait_days)

    - Deterministic (no randomness)
    - Small magnitude (so it won't overpower real model signals)
    """
    prices = [float(p["predicted_price"]) for p in points]
    if not _is_curve_flat(prices):
        return points, False, {}

    rate = _smoothing_rate(dist_class, domestic)

    # Cap per-day absolute drift so cheap routes don't move too little, expensive don't move too much
    # We implement caps by clamping the additive delta, not the rate.
    new_points: List[Dict[str, Any]] = []
    for p in points:
        t = int(p["wait_days"])
        base_price = float(p["predicted_price"])

        delta = base_price * rate * t

        # absolute caps
        delta = float(np.clip(delta, 0.05 * t, 5.0 * t))  # min 5 cents/day, max 5 EUR/day
        adjusted = base_price + delta

        p2 = dict(p)
        p2["predicted_price"] = float(adjusted)
        p2["curve_smoothing_applied"] = True
        new_points.append(p2)

    info = {
        "reason": "flat_curve_detected",
        "rate_fraction_per_day": float(rate),
        "absolute_delta_cap_per_day": {"min": 0.05, "max": 5.0},
    }
    return new_points, True, info


# -----------------------------
# Public API
# -----------------------------
def predict_price(origin: str, destination: str, date_str: str) -> Dict[str, Any]:
    raw, days_left_signed = _build_feature_row(origin, destination, date_str)

    if days_left_signed < 0:
        raise ValueError("Travel date is in the past. Please choose a future date (YYYY-MM-DD).")

    X = _prepare_for_model1(raw, ART.meta)
    pred = ART.pipeline.predict(X)

    price_raw = float(pred[0])
    if price_raw < 0:
        price_raw = 0.0

    price_calib = _apply_calibration(price_raw)
    if price_calib < 0:
        price_calib = 0.0

    return {
        "predicted_price": float(price_calib),          # calibrated
        "predicted_price_raw": float(price_raw),        # raw model output
        "calibration_loaded": bool(_load_calibration()),
        "currency": "EUR",
        "days_left": int(max(days_left_signed, 0)),
        "debug": {
            "origin": str(origin).strip().upper(),
            "destination": str(destination).strip().upper(),
            "date": date_str,
            "date_search": str(raw.loc[0, "date_search"]),
        },
    }


def predict_price_curve(origin: str, destination: str, date_str: str, *, max_wait_days: int = 7) -> Dict[str, Any]:
    """
    Simulate buying later: wait_days = 0..max_wait_days
    - date_search becomes today + t
    - Days_left becomes original_days_left - t
    Uses CALIBRATED prices so Model 2 decisions are consistent.
    Adds a flat-curve smoothing overlay when model outputs are piecewise constant.
    """
    travel_date = _parse_date(date_str)
    today = date.today()
    original_days_left = (travel_date - today).days

    if original_days_left < 0:
        raise ValueError("Travel date is in the past. Please choose a future date (YYYY-MM-DD).")

    points: list[dict] = []
    dist_class_for_curve = "short"
    domestic_for_curve = 0

    for t in range(0, int(max_wait_days) + 1):
        sim_today = today + timedelta(days=t)

        raw, dleft_signed = _build_feature_row(
            origin,
            destination,
            date_str,
            search_date_override=sim_today,
        )

        # keep these for smoothing decision
        if t == 0:
            dist_class_for_curve = str(raw.loc[0, "distance_class"])
            domestic_for_curve = int(raw.loc[0, "domestic"]) if pd.notna(raw.loc[0, "domestic"]) else 0

        X = _prepare_for_model1(raw, ART.meta)
        pred = ART.pipeline.predict(X)

        price_raw = float(pred[0])
        if price_raw < 0:
            price_raw = 0.0

        price_calib = _apply_calibration(price_raw)
        if price_calib < 0:
            price_calib = 0.0

        points.append({
            "wait_days": int(t),
            "days_left_after_wait": int(max(dleft_signed, 0)),
            "predicted_price": float(price_calib),      # calibrated
            "predicted_price_raw": float(price_raw),    # raw
        })

        if dleft_signed <= 0 and t < max_wait_days:
            break

    # ---- Apply flat-curve fix (only if needed) ----
    points2, smoothing_applied, smoothing_info = _apply_curve_smoothing(
        points,
        dist_class=dist_class_for_curve,
        domestic=domestic_for_curve,
    )

    prices_only = [float(p["predicted_price"]) for p in points2]
    best_idx = int(np.argmin(prices_only))
    best_wait_days = int(points2[best_idx]["wait_days"])
    best_price = float(points2[best_idx]["predicted_price"])
    now_price = float(points2[0]["predicted_price"])
    savings_vs_now = float(now_price - best_price)

    eps = 1e-6
    last_price = float(points2[-1]["predicted_price"])
    if last_price > now_price + eps:
        trend_direction = "up"
    elif last_price < now_price - eps:
        trend_direction = "down"
    else:
        trend_direction = "flat"

    return {
        "prices": points2,
        "calibration_loaded": bool(_load_calibration()),
        "curve_smoothing_applied": bool(smoothing_applied),
        "curve_smoothing_info": smoothing_info,
        "summary": {
            "original_days_left": int(original_days_left),
            "best_wait_days": best_wait_days,
            "best_price": best_price,
            "price_now": now_price,
            "savings_vs_now": savings_vs_now,
            "trend_direction": trend_direction,
        },
    }
