# ============================================
# Modell 1 (FAST): Flugpreis-Vorhersage (API-ready)
# - AUTO Features (ohne Preis-Leakage)
# - Global sampling -> Route sampling (viel schneller)
# - OneHotEncoder: min_frequency + max_categories (kein Feature-Explosion)
# - ExtraTrees (schneller als RandomForest)
# ============================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import ExtraTreesRegressor  # schneller
# from sklearn.ensemble import RandomForestRegressor  # falls du zurück willst


# -----------------------------
# 0) SETTINGS
# -----------------------------
CSV_PATH = "master_training_dataset.csv"
TARGET = "Price_num"

# Speed: Erst global reduzieren, dann pro Route balancieren
USE_GLOBAL_SAMPLING = True
GLOBAL_SAMPLE_N = 300_000   # <- für Laptop ok; wenn noch zu langsam: 600_000

USE_ROUTE_SAMPLING = True
N_PER_ROUTE = 300             # <- 200–500 ist realistisch. 2000 ist oft viel zu groß.

# OneHotEncoder (gegen "zu viele Spalten")
# Kategorien, die seltener vorkommen, werden automatisch als "infrequent" gebündelt
OHE_MIN_FREQUENCY = 80        # <- 50–200 (höher = weniger Spalten, schneller)
OHE_MAX_CATEGORIES = 2000     # <- cap pro Spalte (wichtig!)

# Modell: ExtraTrees ist oft schneller als RandomForest
MODEL_PARAMS = dict(
    n_estimators=80,         # <- weniger = schneller; 120–250 ok
    max_depth=14,             # <- begrenzt Laufzeit stark
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

# -----------------------------
# 1) LOAD
# -----------------------------
df = pd.read_csv(CSV_PATH, low_memory=False)
print("Roh-Datensatz:", df.shape)

if TARGET not in df.columns:
    raise ValueError(f"Spalte '{TARGET}' nicht gefunden!")

# -----------------------------
# 2) CLEAN TARGET
# -----------------------------
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])
df = df[df[TARGET] > 0]

# IQR-Ausreißer entfernen
Q1, Q3 = df[TARGET].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df = df[(df[TARGET] >= lower) & (df[TARGET] <= upper)]
print("Bereinigt:", df.shape)

# -----------------------------
# 3) GLOBAL SAMPLING (VOR Route-Sampling = Speed!)
# -----------------------------
if USE_GLOBAL_SAMPLING and len(df) > GLOBAL_SAMPLE_N:
    df = df.sample(GLOBAL_SAMPLE_N, random_state=42).reset_index(drop=True)
    print("Nach Global-Sampling:", df.shape)

# -----------------------------
# 4) ROUTE COLUMN + ROUTE SAMPLING
# -----------------------------
if USE_ROUTE_SAMPLING:
    if "origin_iata" in df.columns and "destination_iata" in df.columns:
        df["route"] = df["origin_iata"].astype(str) + "_" + df["destination_iata"].astype(str)
    else:
        print("WARNUNG: origin_iata / destination_iata fehlen -> Route-Sampling deaktiviert.")
        USE_ROUTE_SAMPLING = False

if USE_ROUTE_SAMPLING:
    # group_keys=False schneller/sauberer, und df ist schon global gesampled
    df = (
        df.groupby("route", group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), N_PER_ROUTE), random_state=42))
          .reset_index(drop=True)
    )
    print("Nach Route-Sampling:", df.shape)

# -----------------------------
# 5) AUTO FEATURE SELECTION (ohne Preis-Leakage)
# -----------------------------
def is_price_leak(col: str) -> bool:
    c = col.lower().strip()
    if c == TARGET.lower():
        return False
    return ("price" in c) or ("fare" in c) or ("cost" in c)

def is_useless_id(col: str) -> bool:
    c = col.lower().strip()
    return c in {"id", "index", "unnamed: 0"} or c.endswith("_id")

excluded_cols = {TARGET}
for col in df.columns:
    if col == TARGET:
        continue
    if is_price_leak(col) or is_useless_id(col):
        excluded_cols.add(col)

feature_cols = [c for c in df.columns if c not in excluded_cols]

# split numeric vs categorical
numeric_cols, categorical_cols = [], []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

# robust numeric conversion (für gemischte Spalten)
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

for c in categorical_cols:
    df[c] = df[c].astype(str).fillna("Unknown")

print("\n--- FEATURES (AUTO) ---")
print("Anzahl Features gesamt:", len(feature_cols))
print("Numerisch:", len(numeric_cols), "| Kategorial:", len(categorical_cols))
print("Beispiel numerisch:", numeric_cols[:12])
print("Beispiel kategorial:", categorical_cols[:12])

# -----------------------------
# 6) BUILD X / y
# -----------------------------
X = df[feature_cols]
y = df[TARGET]

# -----------------------------
# 7) PIPELINE
# -----------------------------
ohe = OneHotEncoder(
    handle_unknown="infrequent_if_exist",   # wichtig!
    min_frequency=OHE_MIN_FREQUENCY,
    max_categories=OHE_MAX_CATEGORIES,
    sparse_output=True
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", ohe, categorical_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("reg", ExtraTreesRegressor(**MODEL_PARAMS))
    # ("reg", RandomForestRegressor(**MODEL_PARAMS))  # alternative
])

# -----------------------------
# 8) TRAIN / TEST
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain/Test:", X_train.shape, X_test.shape)
print("\nTraining startet...")
model.fit(X_train, y_train)
print("Training fertig!")

# -----------------------------
# 9) EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== MODELL-LEISTUNG ===")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 3))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.2)
minv = min(float(y_test.min()), float(np.min(y_pred)))
maxv = max(float(y_test.max()), float(np.max(y_pred)))
plt.plot([minv, maxv], [minv, maxv], "r--")
plt.xlabel("Tatsächlicher Preis")
plt.ylabel("Vorhergesagter Preis")
plt.title("Modell 1 (FAST): Tatsächlich vs. Vorhergesagt")
plt.tight_layout()
plt.show()

# -----------------------------
# 10) SAVE
# -----------------------------
joblib.dump(model, "model1_pipeline_fast.pkl")

meta = {
    "csv_path": CSV_PATH,
    "target": TARGET,
    "feature_cols": feature_cols,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "excluded_cols": sorted(list(excluded_cols)),
    "ohe_min_frequency": OHE_MIN_FREQUENCY,
    "ohe_max_categories": OHE_MAX_CATEGORIES,
    "model_params": MODEL_PARAMS,
    "route_sampling": USE_ROUTE_SAMPLING,
    "n_per_route": N_PER_ROUTE,
    "global_sampling": USE_GLOBAL_SAMPLING,
    "global_sample_n": GLOBAL_SAMPLE_N,
}
joblib.dump(meta, "model1_metadata_fast.pkl")

print("\n✅ Gespeichert:")
print("- model1_pipeline_fast.pkl")
print("- model1_metadata_fast.pkl")
