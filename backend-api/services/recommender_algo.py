# services/recommender_algo.py
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import math
import csv

# Try pandas (fast). If not available, fallback to csv module aggregation.
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


# -----------------------------
# Paths
# -----------------------------
# This matches your backend structure used in model1_service.py:
# BASE_DIR = backend_api/
# routes file expected at: backend_api/data/routes_features_final_continents_filled.csv
BASE_DIR = Path(__file__).resolve().parents[1]
ROUTES_CSV = BASE_DIR / "data" / "routes_features_final_continents_filled.csv"


# -----------------------------
# Caches
# -----------------------------
# origin -> list of candidate destination rows
_ORIGIN_INDEX: Dict[str, List[Dict[str, Any]]] = {}
# destination -> metadata (name/country/continent)
_DEST_META: Dict[str, Dict[str, Any]] = {}
_LOADED = False


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return float(v) if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _norm(x: float, maxv: float) -> float:
    if maxv <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, x / maxv))


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def budget_bucket(price: float) -> str:
    if price < 120:
        return "cheap"
    if price < 250:
        return "mid"
    return "high"


def _get_dest_meta(dest: str) -> Dict[str, Any]:
    d = dest.upper().strip()
    return _DEST_META.get(d, {"destination_name": "", "country": "", "continent": ""})


def _load_routes_once() -> None:
    """
    Loads routes_features_final_continents_filled.csv and builds:
      - _ORIGIN_INDEX: origin -> list of {destination, distance_km, ...}
      - _DEST_META: destination -> {destination_name, country, continent}
    """
    global _LOADED, _ORIGIN_INDEX, _DEST_META
    if _LOADED:
        return

    _ORIGIN_INDEX = {}
    _DEST_META = {}

    if not ROUTES_CSV.exists():
        # If this happens, you placed the file in the wrong folder.
        # Put it at: backend_big/data/routes_features_final_continents_filled.csv
        _LOADED = True
        return

    if pd is not None:
        df = pd.read_csv(ROUTES_CSV, low_memory=False)

        # normalize column names for safety
        cols = {c.lower(): c for c in df.columns}

        def col(name: str) -> Optional[str]:
            return cols.get(name.lower())

        # Required columns (flexible)
        origin_col = col("origin_iata")
        dest_col = col("destination_iata")
        if not origin_col or not dest_col:
            raise ValueError(
                "Routes file missing origin_iata / destination_iata columns (or different naming)."
            )

        # optional meta columns
        name_col = col("dest_name")
        country_col = col("dest_iso_country")
        cont_col = col("dest_continent")

        # optional numeric/feature columns (use if present)
        dist_col = col("distance_km")
        dist_class_col = col("distance_class")
        domestic_col = col("domestic")
        pop_col = col("route_popularity")
        comp_col = col("competition_score")
        hub_col = col("dest_hub_score")
        conn_col = col("dest_connectivity")

        # build destination metadata
        meta_cols = [dest_col]
        if name_col:
            meta_cols.append(name_col)
        if country_col:
            meta_cols.append(country_col)
        if cont_col:
            meta_cols.append(cont_col)

        meta_df = df[meta_cols].dropna(subset=[dest_col]).drop_duplicates(subset=[dest_col])
        for _, r in meta_df.iterrows():
            d = str(r[dest_col]).upper().strip()
            _DEST_META[d] = {
                "destination_name": str(r[name_col]) if name_col else "",
                "country": str(r[country_col]) if country_col else "",
                "continent": str(r[cont_col]) if cont_col else "",
            }

        # aggregate per origin->destination
        # if a col doesn't exist, create safe defaults
        def get_series(cname: Optional[str], default_val: Any):
            if cname and cname in df.columns:
                return df[cname]
            return default_val

        df2 = df.copy()
        df2[origin_col] = df2[origin_col].astype(str).str.upper().str.strip()
        df2[dest_col] = df2[dest_col].astype(str).str.upper().str.strip()

        # ensure columns exist
        if dist_col is None:
            df2["_distance_km"] = 0.0
            dist_col = "_distance_km"
        if dist_class_col is None:
            df2["_distance_class"] = ""
            dist_class_col = "_distance_class"
        if domestic_col is None:
            df2["_domestic"] = 0
            domestic_col = "_domestic"
        if pop_col is None:
            df2["_route_popularity"] = 0.0
            pop_col = "_route_popularity"
        if comp_col is None:
            df2["_competition_score"] = 0.0
            comp_col = "_competition_score"
        if hub_col is None:
            df2["_dest_hub_score"] = 0.0
            hub_col = "_dest_hub_score"
        if conn_col is None:
            df2["_dest_connectivity"] = 0.0
            conn_col = "_dest_connectivity"

        agg = (
            df2.groupby([origin_col, dest_col], as_index=False)
            .agg(
                distance_km=(dist_col, "mean"),
                distance_class=(dist_class_col, "first"),
                domestic=(domestic_col, "max"),
                route_popularity=(pop_col, "max"),
                competition_score=(comp_col, "max"),
                dest_hub_score=(hub_col, "max"),
                dest_connectivity=(conn_col, "max"),
            )
        )

        for _, r in agg.iterrows():
            o = str(r[origin_col]).upper().strip()
            d = str(r[dest_col]).upper().strip()

            row = {
                "origin": o,
                "destination": d,
                "distance_km": _safe_float(r.get("distance_km", 0.0)),
                "distance_class": str(r.get("distance_class", "") or ""),
                "domestic": int(_safe_float(r.get("domestic", 0))),
                "route_popularity": _safe_float(r.get("route_popularity", 0.0)),
                "competition_score": _safe_float(r.get("competition_score", 0.0)),
                "dest_hub_score": _safe_float(r.get("dest_hub_score", 0.0)),
                "dest_connectivity": _safe_float(r.get("dest_connectivity", 0.0)),
            }
            _ORIGIN_INDEX.setdefault(o, []).append(row)

        _LOADED = True
        return

    # --------- csv fallback (no pandas) ----------
    # Aggregate route features per (origin, destination)
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with ROUTES_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            o = (r.get("origin_iata") or "").upper().strip()
            d = (r.get("destination_iata") or "").upper().strip()
            if not o or not d:
                continue

            if d not in _DEST_META:
                _DEST_META[d] = {
                    "destination_name": (r.get("dest_name") or ""),
                    "country": (r.get("dest_iso_country") or ""),
                    "continent": (r.get("dest_continent") or ""),
                }

            key = (o, d)
            if key not in best:
                best[key] = {
                    "origin": o,
                    "destination": d,
                    "distance_km": _safe_float(r.get("distance_km", 0.0)),
                    "distance_class": (r.get("distance_class") or ""),
                    "domestic": int(_safe_float(r.get("domestic", 0))),
                    "route_popularity": _safe_float(r.get("route_popularity", 0.0)),
                    "competition_score": _safe_float(r.get("competition_score", 0.0)),
                    "dest_hub_score": _safe_float(r.get("dest_hub_score", 0.0)),
                    "dest_connectivity": _safe_float(r.get("dest_connectivity", 0.0)),
                    "_n": 1,
                }
            else:
                cur = best[key]
                cur["_n"] += 1
                # mean distance
                cur["distance_km"] = (cur["distance_km"] * (cur["_n"] - 1) + _safe_float(r.get("distance_km", 0.0))) / cur["_n"]
                # max for others
                for c in ("route_popularity", "competition_score", "dest_hub_score", "dest_connectivity", "domestic"):
                    cur[c] = max(cur[c], _safe_float(r.get(c, 0.0)))

    for (o, _), row in best.items():
        row.pop("_n", None)
        _ORIGIN_INDEX.setdefault(o, []).append(row)

    _LOADED = True


def build_user_profile(searches: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
    """
    Profile from DB searches:
      - preferred continent (based on past destinations)
      - preferred distance_class (if present in route index)
      - preferred budget bucket
    """
    _load_routes_once()

    user_rows = [s for s in searches if str(s.get("user_id")) == str(user_id)]
    if not user_rows:
        return {"continent": None, "distance_class": None, "bucket": None}

    dests = [str(r.get("destination", "")).upper().strip() for r in user_rows if r.get("destination")]
    conts = []
    for d in dests:
        cont = str(_get_dest_meta(d).get("continent") or "").strip()
        if cont:
            conts.append(cont)

    prices = [float(r.get("price", 0.0) or 0.0) for r in user_rows]
    bucket = Counter([budget_bucket(p) for p in prices]).most_common(1)[0][0] if prices else None
    continent = Counter(conts).most_common(1)[0][0] if conts else None

    # distance_class preference: look up route row if possible
    dist_classes = []
    for r in user_rows:
        o = str(r.get("origin", "")).upper().strip()
        d = str(r.get("destination", "")).upper().strip()
        if o and d and o in _ORIGIN_INDEX:
            match = next((x for x in _ORIGIN_INDEX[o] if x["destination"] == d), None)
            if match and match.get("distance_class"):
                dist_classes.append(str(match["distance_class"]))
    distance_class = Counter(dist_classes).most_common(1)[0][0] if dist_classes else None

    return {"continent": continent, "distance_class": distance_class, "bucket": bucket}


def _estimate_price(
    *,
    base_price: float,
    cand_distance: float,
    ref_distance: float,
    competition_norm: float,
    hub_norm: float,
) -> float:
    """
    Explainable price proxy:
      base_price * (distance_ratio^0.8) * small discounts for competition/hub
    """
    base_price = max(20.0, float(base_price))
    ref_distance = max(200.0, float(ref_distance))
    cand_distance = max(200.0, float(cand_distance))

    ratio = cand_distance / ref_distance
    price = base_price * (ratio ** 0.8)

    price *= (1.0 - 0.06 * max(0.0, min(1.0, competition_norm)))
    price *= (1.0 - 0.04 * max(0.0, min(1.0, hub_norm)))

    return float(max(25.0, min(price, base_price * 2.8)))


def recommend_hybrid(
    *,
    user_id: str,
    origin: str,
    month: int,
    price: float,
    k: int,
    searches: List[Dict[str, Any]],
    feedback: Dict[str, int],
    popularity: Dict[str, int],
    exclude_destination: str | None = None,
    reference_destination: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Diverse hybrid recommender using REAL candidates from routes dataset.
    """
    _load_routes_once()

    origin = origin.upper().strip()
    exclude = exclude_destination.upper().strip() if exclude_destination else None
    ref_dest = reference_destination.upper().strip() if reference_destination else exclude

    candidates = list(_ORIGIN_INDEX.get(origin, []))
    if not candidates:
        return []

    # filter
    filtered = []
    for c in candidates:
        d = c["destination"]
        if d == origin:
            continue
        if exclude and d == exclude:
            continue
        filtered.append(c)
    candidates = filtered

    if not candidates:
        return []

    query_bucket = budget_bucket(float(price))
    profile = build_user_profile(searches, user_id)

    # reference distance: distance to reference destination if available, else median distance
    ref_distance = None
    if ref_dest:
        m = next((x for x in candidates if x["destination"] == ref_dest), None)
        if m:
            ref_distance = float(m.get("distance_km", 0.0) or 0.0)

    if not ref_distance or ref_distance <= 0:
        ds = sorted([_safe_float(x.get("distance_km", 0.0)) for x in candidates if _safe_float(x.get("distance_km", 0.0)) > 0])
        ref_distance = ds[len(ds) // 2] if ds else 1000.0

    max_pop = max([_safe_float(c.get("route_popularity", 0.0)) for c in candidates] + [1.0])
    max_hub = max([_safe_float(c.get("dest_hub_score", 0.0)) for c in candidates] + [1.0])
    max_comp = max([_safe_float(c.get("competition_score", 0.0)) for c in candidates] + [1.0])
    max_conn = max([_safe_float(c.get("dest_connectivity", 0.0)) for c in candidates] + [1.0])
    max_db_pop = max(list(popularity.values()) + [1])

    scored: List[Dict[str, Any]] = []

    for c in candidates:
        dest = c["destination"]
        meta = _get_dest_meta(dest)

        rp = _norm(_safe_float(c.get("route_popularity", 0.0)), max_pop)
        hub = _norm(_safe_float(c.get("dest_hub_score", 0.0)), max_hub)
        comp = _norm(_safe_float(c.get("competition_score", 0.0)), max_comp)
        conn = _norm(_safe_float(c.get("dest_connectivity", 0.0)), max_conn)

        db_pop = popularity.get(dest, 0)
        db_pop_norm = _norm(float(db_pop), float(max_db_pop))

        fb = int(feedback.get(dest, 0))  # +1/-1/0

        score = 0.0
        why_tags: List[str] = []

        # Route/network strength (real intelligence)
        score += 1.40 * rp
        score += 0.90 * hub
        score += 0.55 * comp
        score += 0.35 * conn
        why_tags.append("route_strength")

        # Personalization
        if profile.get("continent") and meta.get("continent") == profile["continent"]:
            score += 0.60
            why_tags.append("matches_continent")
        if profile.get("distance_class") and str(c.get("distance_class", "")) == profile["distance_class"]:
            score += 0.35
            why_tags.append("matches_distance")
        if profile.get("bucket") and profile.get("bucket") == query_bucket:
            score += 0.20
            why_tags.append("matches_budget")

        # Popularity + exploration
        score += 0.15 * db_pop_norm
        score += 0.20 * (1.0 - db_pop_norm)
        why_tags.append("popularity_exploration")

        # Feedback (strongest)
        if fb != 0:
            score += fb * 1.10
            why_tags.append(f"feedback_{fb:+d}")

        pred_price = _estimate_price(
            base_price=float(price),
            cand_distance=float(c.get("distance_km", 0.0) or 0.0),
            ref_distance=float(ref_distance),
            competition_norm=comp,
            hub_norm=hub,
        )

        scored.append({
            "destination": dest,
            "destination_name": meta.get("destination_name", ""),
            "country": meta.get("country", ""),
            "continent": meta.get("continent", ""),
            "distance_km": float(round(_safe_float(c.get("distance_km", 0.0)), 1)),
            "distance_class": str(c.get("distance_class", "") or ""),
            "predicted_price": float(round(pred_price, 2)),
            "score_raw": float(score),
            "why_tags": why_tags,
        })

    scored.sort(key=lambda x: x["score_raw"], reverse=True)

    # -----------------------------
    # Diversity re-ranking (greedy)
    # -----------------------------
    selected: List[Dict[str, Any]] = []
    used_cont = Counter()
    used_country = Counter()
    used_dist = Counter()

    def diversity_penalty(item: Dict[str, Any]) -> float:
        pen = 0.0
        cont = (item.get("continent") or "").strip()
        country = (item.get("country") or "").strip()
        distc = (item.get("distance_class") or "").strip()

        if country and used_country[country] > 0:
            pen += 0.55
        if cont and used_cont[cont] > 0:
            pen += 0.25
        if distc and used_dist[distc] > 0:
            pen += 0.12
        return pen

    pool = scored[:500]
    target_k = max(1, int(k))

    while pool and len(selected) < target_k:
        best_i = 0
        best_val = -1e9
        for i, it in enumerate(pool):
            val = float(it["score_raw"]) - diversity_penalty(it)
            if val > best_val:
                best_val = val
                best_i = i

        pick = pool.pop(best_i)
        pick["score"] = float(round(best_val, 4))

        # human readable "why"
        loc_parts = []
        if pick.get("continent"):
            loc_parts.append(str(pick["continent"]))
        if pick.get("country"):
            loc_parts.append(str(pick["country"]))
        if pick.get("distance_class"):
            loc_parts.append(str(pick["distance_class"]))
        loc = " • ".join([p for p in loc_parts if p])

        pick["why"] = (
            f"{pick['destination']} ({pick.get('destination_name','')}) — {loc}. "
            f"Estimated ~{pick['predicted_price']:.2f} EUR."
        ).strip()

        selected.append(pick)

        cont = (pick.get("continent") or "").strip()
        country = (pick.get("country") or "").strip()
        distc = (pick.get("distance_class") or "").strip()
        if cont:
            used_cont[cont] += 1
        if country:
            used_country[country] += 1
        if distc:
            used_dist[distc] += 1

    # Return cleaned rows
    out: List[Dict[str, Any]] = []
    for r in selected:
        out.append({
            "destination": r["destination"],
            "destination_name": r.get("destination_name", ""),
            "country": r.get("country", ""),
            "continent": r.get("continent", ""),
            "distance_km": r.get("distance_km", None),
            "distance_class": r.get("distance_class", ""),
            "predicted_price": r.get("predicted_price", None),
            "score": r.get("score", r.get("score_raw", 0.0)),
            "why": r.get("why", ""),
            "why_tags": r.get("why_tags", []),
        })
    return out
