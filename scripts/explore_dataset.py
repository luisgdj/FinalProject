import pandas as pd
from pathlib import Path

# Load dataset
dataset_path = Path("../data/raw")
csv_file = list(dataset_path.glob("*.csv"))[0]
df = pd.read_csv(csv_file)

print(f"\nDataset: {csv_file.name}")
print(f" - Rows: {len(df)}")
print(f" - Columns: {len(df.columns)}\n")

# Display columns
print("Available columns:")
print(df.columns.tolist())

# First rows
print("\nFirst rows:")
print(df.head(3))

# Basic statistics
print("\nInfo:")
print(df.info())
