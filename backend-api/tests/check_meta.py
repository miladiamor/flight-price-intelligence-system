import joblib
from pathlib import Path

META = Path("models") / "model1_metadata_fast.pkl"
meta = joblib.load(META)

print("\n=== META FILE ===")
print("Path:", META.resolve())

print("\n=== COUNTS ===")
print("feature_cols:", len(meta.get("feature_cols", [])))
print("numeric_cols :", len(meta.get("numeric_cols", [])))
print("categorical_cols:", len(meta.get("categorical_cols", [])))

print("\n=== FIRST 120 feature_cols ===")
for c in meta["feature_cols"][:120]:
    print(c)

# helpful: search for important keywords
keywords = ["distance", "km", "duration", "stop", "airline", "route", "hub", "continent", "lat", "lon", "day", "month"]
print("\n=== KEYWORD CHECK ===")
for k in keywords:
    hits = [c for c in meta["feature_cols"] if k in c.lower()]
    print(f"{k:10s} -> {len(hits)} hits")
    if hits[:10]:
        print("  example:", hits[:10])
