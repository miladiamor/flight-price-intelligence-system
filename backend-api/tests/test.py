from services.model1_service import predict_price

tests = [
    ("BER", "CDG", "2026-01-15"),
    ("BER", "JFK", "2026-01-15"),
    ("BER", "HND", "2026-01-15"),
    ("BER", "MAD", "2026-01-15"),
]

for o, d, date in tests:
    print(o, d, predict_price(o, d, date))
import joblib
import numpy as np

model = joblib.load("models/model1_pipeline_fast.pkl")

reg = model.named_steps["reg"]
pre = model.named_steps["preprocess"]

# Get feature names after preprocessing
feature_names = pre.get_feature_names_out()

importances = reg.feature_importances_
pairs = sorted(
    zip(importances, feature_names),
    reverse=True
)[:20]

for imp, name in pairs:
    print(f"{imp:.4f}  {name}")
