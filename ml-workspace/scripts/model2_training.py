# ============================================
# Model 2 (Advisor) - basiert auf Model 1
# - lädt model1_pipeline_fast.pkl + model1_metadata_fast.pkl
# - bereitet Features robust vor (fehlende Spalten, NaN, route)
# - macht zwei Prognosen: jetzt vs. nach WAIT_DAYS Tagen
# - gibt Empfehlung BUY_NOW / WAIT
# ============================================

import joblib
import numpy as np
import pandas as pd

# -----------------------------
# SETTINGS
# -----------------------------
CSV_PATH   = "master_training_dataset.csv"

MODEL1_PATH = "model1_pipeline_fast.pkl"
META1_PATH  = "model1_metadata_fast.pkl"

SAMPLE_N = 200_000          # kleiner machen wenn Laptop langsam ist
WAIT_DAYS = 7               # "warte 7 Tage" Szenario
THRESHOLD_EUR = 10          # min. Preisvorteil in € damit WAIT sinnvoll ist
THRESHOLD_PCT = 0.03        # oder 3% Unterschied

OUT_PATH = "model2_advice_output.csv"


# -----------------------------
# Helper: Feature-Prep für Model1
# -----------------------------
def prepare_for_model1(df_raw: pd.DataFrame, meta: dict) -> pd.DataFrame:
    feature_cols = meta["feature_cols"]
    numeric_cols = meta["numeric_cols"]
    categorical_cols = meta["categorical_cols"]

    df = df_raw.copy()

    # 1) route erzeugen falls Model1 sie erwartet
    if "route" in feature_cols and "route" not in df.columns:
        if "origin_iata" in df.columns and "destination_iata" in df.columns:
            df["route"] = df["origin_iata"].astype(str) + "_" + df["destination_iata"].astype(str)
        else:
            # wenn route erwartet wird aber IATA fehlt -> dummy (wird durch Encoder zu unknown)
            df["route"] = "UNKNOWN_UNKNOWN"

    # 2) Fehlende Feature-Spalten hinzufügen
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 3) Typen + Missing Values fixen
    # numeric: to_numeric + median fill (aus dem aktuellen df)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[col].median()
        if pd.isna(med):
            med = 0.0
        df[col] = df[col].fillna(med)

    # categorical: str + Unknown
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].replace(["nan", "None", "NaT"], "Unknown")
        df[col] = df[col].fillna("Unknown")

    return df[feature_cols]


def shift_time_simple(X: pd.DataFrame, wait_days: int) -> pd.DataFrame:
    """
    Minimaler "Warten"-Counterfactual:
    - Days_left wird reduziert
    Optional kannst du auch Day_of_week/Days_into_month etc. anpassen,
    aber schon Days_left allein gibt oft einen brauchbaren Trend.
    """
    X2 = X.copy()
    if "Days_left" in X2.columns:
        X2["Days_left"] = pd.to_numeric(X2["Days_left"], errors="coerce").fillna(0)
        X2["Days_left"] = (X2["Days_left"] - wait_days).clip(lower=0)
    return X2


# -----------------------------
# MAIN
# -----------------------------
print("✅ Lade Model 1...")
model1 = joblib.load(MODEL1_PATH)
meta1  = joblib.load(META1_PATH)

print("✅ Lade Daten...")
df = pd.read_csv(CSV_PATH, low_memory=False)

# Sample (damit es schnell bleibt)
if SAMPLE_N and len(df) > SAMPLE_N:
    df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)

print("Daten-Shape:", df.shape)

# Features für Model1 bauen (NaN/fehlende Spalten beheben)
X_now = prepare_for_model1(df, meta1)

# Preis jetzt (falls vorhanden – nur für Reporting)
current_price = None
if "Price_num" in df.columns:
    current_price = pd.to_numeric(df["Price_num"], errors="coerce")

# Prognose jetzt
pred_now = model1.predict(X_now)

# Prognose nach WAIT_DAYS (vereinfachtes Counterfactual)
X_wait = shift_time_simple(X_now, WAIT_DAYS)
pred_wait = model1.predict(X_wait)

delta = pred_wait - pred_now

# Entscheidung:
# Wenn Preis später deutlich höher -> BUY_NOW
# Wenn Preis später nicht höher (oder niedriger) -> WAIT
buy_now = (pred_wait > (pred_now * (1 + THRESHOLD_PCT) + THRESHOLD_EUR))
decision = np.where(buy_now, "BUY_NOW", "WAIT")

out = pd.DataFrame({
    "pred_price_now": pred_now,
    "pred_price_wait": pred_wait,
    "delta_wait_minus_now": delta,
    "decision": decision
})

if current_price is not None:
    out.insert(0, "current_price_observed", current_price)

print("\n=== Model 2 (Advisor) Summary ===")
print(out["decision"].value_counts(dropna=False))

out.to_csv(OUT_PATH, index=False)
print(f"\n✅ Gespeichert: {OUT_PATH}")
