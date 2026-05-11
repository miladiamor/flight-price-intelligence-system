import pandas as pd
import os

input_file = "master_training_dataset_enriched_v2_continents_filled.csv"
output_prefix = "master_training_dataset_part"
chunksize = 426258  # ~4.26M / 10

os.makedirs("parts", exist_ok=True)

for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize), start=1):
    output_file = f"parts/{output_prefix}_{i}.csv"
    chunk.to_csv(output_file, index=False)
    print(f"Saved {output_file}")

print("Done! CSV split into 10 parts.")
