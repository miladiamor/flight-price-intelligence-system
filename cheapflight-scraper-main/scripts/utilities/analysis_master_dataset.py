import pandas as pd

# path to your full dataset
df = pd.read_csv("master_training_dataset_enriched_v2_fixed.csv", low_memory=False)

# 1. first rows
print("HEAD:")
print(df.head())

# 2. basic info: columns, dtypes, non-null counts
print("\nINFO:")
print(df.info())

# 3. numeric summary (price, days_left, etc.)
print("\nDESCRIBE NUMERIC:")
print(df.describe())

# 4. how many missing values per column (sorted)
print("\nMISSING VALUES (% of rows):")
missing = df.isna().mean().sort_values(ascending=False)
print(missing.head(30))   # top 30 most-missing columns

# 5. random sample of 5 rows to eyeball
print("\nRANDOM SAMPLE:")
print(df.sample(5, random_state=42))
