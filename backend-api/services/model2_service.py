# services/model2_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math

import numpy as np

from services.model1_service import predict_price_curve


# -----------------------------
# Config / thresholds
# -----------------------------
@dataclass(frozen=True)
class AdvisorConfig:
    # Compute a longer horizon so WAIT is actually possible even if the next 7 days are monotonic.
    # UI can still display only the first 7 days.
    max_wait_days: int = 30

    # Savings thresholds (dynamic)
    min_abs_savings_floor: float = 2.0      # at least 2 EUR
    min_pct_savings_floor: float = 0.03     # at least 3%
    strong_pct_savings: float = 0.06        # 6%+ = strong signal

    # Confidence thresholds
    wait_confidence_threshold: float = 0.60
    buy_confidence_threshold: float = 0.55

    # Penalize confidence if curve smoothing applied (fallback curve / synthetic)
    smoothing_conf_penalty: float = 0.20

    # Penalize if curve is too flat/low-variance
    low_var_conf_penalty: float = 0.15

    # If departure is very close, bias toward BUY_NOW
    close_departure_days: int = 7
    close_departure_penalty_wait: float = 0.20


CFG = AdvisorConfig()


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _linear_slope(xs: List[float], ys: List[float]) -> float:
    """Return slope of y ~ a*x + b. If degenerate, slope=0."""
    if len(xs) < 2:
        return 0.0
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.std(x) < 1e-9:
        return 0.0
    a = np.polyfit(x, y, 1)[0]
    return float(a)


def _compute_dynamic_thresholds(now_price: float) -> Tuple[float, float]:
    """
    Dynamic thresholds that work for both cheap and expensive routes.
    - Abs threshold grows with price but has a floor.
    - Percent threshold has a floor.
    """
    now_price = max(now_price, 1e-6)

    # absolute savings: at least 2 EUR OR 3% of price (whichever bigger), capped
    abs_thr = max(CFG.min_abs_savings_floor, 0.03 * now_price)
    abs_thr = min(abs_thr, 35.0)

    pct_thr = CFG.min_pct_savings_floor
    return float(abs_thr), float(pct_thr)


def _confidence_score(
    *,
    now_price: float,
    best_price: float,
    best_wait: int,
    prices: List[float],
    smoothing_applied: bool,
    days_left: int,
) -> Tuple[float, List[str]]:
    """
    Confidence combines:
    - savings strength (for WAIT cases)
    - curve variance (avoid trusting almost-flat curves)
    - proximity to departure
    - smoothing penalty
    """
    reasons: List[str] = []
    now_price = max(now_price, 1e-6)

    savings_abs = now_price - best_price
    savings_pct = savings_abs / now_price

    xs = list(range(len(prices)))
    slope = _linear_slope(xs, prices)
    rng = float(np.max(prices) - np.min(prices)) if len(prices) >= 2 else 0.0

    # Base confidence from savings strength (WAIT cases)
    base = float(np.clip(savings_pct / CFG.strong_pct_savings, 0.0, 1.0))

    # BUY_NOW cases: "0 savings" shouldn't force low confidence.
    # If the curve is not clearly trending down, BUY_NOW can still be confident.
    if best_wait == 0:
        if slope >= 0.02:
            base = max(base, 0.62)
            reasons.append("trend_not_down_supports_buy")
        elif slope <= -0.05:
            base = min(base, 0.45)
            reasons.append("downtrend_warning")
        else:
            base = max(base, 0.55)
            reasons.append("no_strong_trend")

    # Penalize low-variance curves (absolute + relative)
    rel_rng = rng / max(now_price, 1e-6)
    if (rng < 0.75) or (rel_rng < 0.006):  # <€0.75 swing OR <0.6% swing
        base -= CFG.low_var_conf_penalty
        reasons.append("low_variance_curve")

    # Penalize if very close to departure
    if days_left <= CFG.close_departure_days:
        base -= 0.10
        reasons.append("close_to_departure")

    # Penalize smoothing (fallback curve / synthetic curve shaping)
    if smoothing_applied:
        base -= CFG.smoothing_conf_penalty
        reasons.append("curve_smoothing_penalty")

    base = float(np.clip(base, 0.0, 1.0))
    return base, reasons


def _compute_from_curve(
    *,
    curve: Dict[str, Any],
    price_now_override: Optional[float] = None,
    days_left_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Core decision-making from an already computed Model1 curve.
    Returns keys that match your UI:
      decision, price_now, best_price, best_wait_days, expected_savings, reason, ...
    """
    points = curve.get("prices") or []
    if not points:
        raise ValueError("Model1 curve returned no points.")

    # ✅ FIX: Model1 indicates fallback/smoothing via summary.used_fallback_curve.
    summary = curve.get("summary") or {}
    smoothing_applied = bool(
        curve.get("curve_smoothing_applied", False) or
        summary.get("used_fallback_curve", False)
    )

    # now price
    now_curve = _safe_float(points[0].get("predicted_price"))
    now_price = _safe_float(price_now_override) if price_now_override is not None else now_curve
    if not math.isfinite(now_price):
        now_price = now_curve

    # best point
    best_point = min(points, key=lambda p: _safe_float(p.get("predicted_price", 1e18)))
    best_price = _safe_float(best_point.get("predicted_price"))
    best_wait = int(best_point.get("wait_days", 0))

    # days left
    days_left = (
        int(days_left_override)
        if days_left_override is not None
        else int(points[0].get("days_left_after_wait", summary.get("original_days_left", 0)))
    )

    savings_abs = float(now_price - best_price) if (math.isfinite(now_price) and math.isfinite(best_price)) else 0.0
    savings_abs = max(savings_abs, 0.0)
    savings_pct = float(savings_abs / max(now_price, 1e-6))

    abs_thr, pct_thr = _compute_dynamic_thresholds(now_price)

    prices_only = [float(_safe_float(p.get("predicted_price"))) for p in points]

    # Extra: cost of waiting 7 days vs buying now (often what users want to see)
    idx7 = min(7, len(prices_only) - 1)
    cost_of_waiting_7d = float(prices_only[idx7] - now_price)

    conf, conf_reasons = _confidence_score(
        now_price=now_price,
        best_price=best_price,
        best_wait=best_wait,
        prices=prices_only,
        smoothing_applied=smoothing_applied,
        days_left=days_left,
    )

    reasons: List[str] = []
    reasons.extend(conf_reasons)

    decision = "NO_CLEAR_SIGNAL"

    if best_wait == 0:
        # ✅ IMPORTANT FIX:
        # If the minimum within the computed horizon is today, the action is BUY_NOW.
        # Confidence should not flip this to NO_CLEAR_SIGNAL; it can only add an informational tag.
        decision = "BUY_NOW"
        reasons.append("best_price_is_now")
        if conf < CFG.buy_confidence_threshold:
            reasons.append("low_confidence_tag_only")
    else:
        strong_savings = (savings_abs >= abs_thr) and (savings_pct >= pct_thr)

        if days_left <= CFG.close_departure_days:
            if strong_savings and conf >= (CFG.wait_confidence_threshold + CFG.close_departure_penalty_wait):
                decision = "WAIT"
                reasons.append("strong_savings_even_close_to_departure")
            else:
                decision = "BUY_NOW"
                reasons.append("close_to_departure_bias_buy")
        else:
            if strong_savings and conf >= CFG.wait_confidence_threshold:
                decision = "WAIT"
                reasons.append("strong_savings_wait_recommended")
            elif savings_abs <= abs_thr * 0.35:
                decision = "BUY_NOW"
                reasons.append("waiting_not_worth_it")
            else:
                decision = "NO_CLEAR_SIGNAL"
                reasons.append("mixed_or_weak_signal")

    if decision == "BUY_NOW":
        msg = "The cheapest predicted price is today."
    elif decision == "WAIT":
        msg = f"Waiting {best_wait} day(s) is predicted to save money."
    else:
        msg = "Predicted savings from waiting are too small (or uncertain) to be confident."

    return {
        # ---- UI keys expected by static/app.js ----
        "decision": decision,
        "price_now": round(float(now_price), 2) if math.isfinite(now_price) else None,
        "best_price": round(float(best_price), 2) if math.isfinite(best_price) else None,
        "best_wait_days": int(best_wait),
        "expected_savings": round(float(savings_abs), 2),
        "reason": msg,

        # ---- extra diagnostics (optional) ----
        "confidence": float(round(conf, 3)),
        "expected_savings_pct": float(round(savings_pct * 100.0, 2)),
        "cost_of_waiting_7d": float(round(cost_of_waiting_7d, 2)),
        "thresholds": {
            "min_abs_savings_eur": float(round(abs_thr, 2)),
            "min_pct_savings": float(round(pct_thr * 100.0, 2)),
        },
        "meta": {
            "calibration_loaded": bool(curve.get("calibration_loaded", False)),
            "curve_smoothing_applied": bool(smoothing_applied),
            "reasons": reasons[:12],
        },
    }


# -----------------------------
# Public API
# -----------------------------
def advise(*args, **kwargs) -> Dict[str, Any]:
    """
    Supports BOTH calling styles:

    A) advise(origin, destination, date_str)
    B) advise(price_now=..., curve=..., days_left=...)

    Returns UI-compatible dict used by static/app.js.
    """

    # --- Style B: keyword args ---
    if "curve" in kwargs:
        curve = kwargs.get("curve")
        price_now = kwargs.get("price_now", None)
        days_left = kwargs.get("days_left", None)
        if not isinstance(curve, dict):
            raise ValueError("advise(curve=...) must receive a dict.")
        return _compute_from_curve(
            curve=curve,
            price_now_override=_safe_float(price_now) if price_now is not None else None,
            days_left_override=int(days_left) if days_left is not None else None,
        )

    # --- Style A: positional ---
    if len(args) >= 3:
        origin, destination, date_str = args[0], args[1], args[2]
    else:
        origin = kwargs.get("origin")
        destination = kwargs.get("destination")
        date_str = kwargs.get("date") or kwargs.get("date_str")

    if not origin or not destination or not date_str:
        raise ValueError("advise requires (origin, destination, date_str) or (price_now, curve, days_left).")

    curve = predict_price_curve(str(origin), str(destination), str(date_str), max_wait_days=CFG.max_wait_days)
    return _compute_from_curve(curve=curve)
