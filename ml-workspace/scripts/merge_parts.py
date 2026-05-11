import pandas as pd
from pathlib import Path

folder = Path(".")  # aktueller Ordner
parts = sorted(folder.glob("master_training_dataset_part_*.csv"))

if not parts:
    raise FileNotFoundError("Keine master_training_dataset_part_*.csv gefunden!")

dfs = []
for p in parts:
    print("Lese:", p.name)
    dfs.append(pd.read_csv(p, low_memory=False))

# Optional: prüfen, ob alle Spalten gleich sind
cols0 = set(dfs[0].columns)
for i, d in enumerate(dfs[1:], start=2):
    if set(d.columns) != cols0:
        print(f"⚠️ Spalten unterscheiden sich in part_{i}")

df = pd.concat(dfs, ignore_index=True)
out = folder / "master_training_dataset.csv"
df.to_csv(out, index=False)
print("✅ Gespeichert:", out, "Shape:", df.shape)
