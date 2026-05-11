# check_missing_features.py
import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models")
meta = joblib.load(MODELS_DIR / "model1_metadata_fast.pkl")

feature_cols = meta["feature_cols"]

# what your service currently provides:
provided = {"origin_iata","destination_iata","Days_left","Month_of_year","Day_of_week","route"}

missing = [c for c in feature_cols if c not in provided]
print("Total feature_cols:", len(feature_cols))
print("Provided:", len(provided))
print("Missing:", len(missing))
print("\nMissing examples:", missing[:30])
