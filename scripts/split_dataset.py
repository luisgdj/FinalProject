import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/raw/dataset.csv")

print(f"Original dataset: {len(df)} rows")

# Drop rows with null values in key columns
df = df.dropna(subset=['smiles', 'InChIKEY', 'CCS_AVG', 'Adduct'])
print(f"After cleaning: {len(df)} rows")

# Remove dimers
df = df[df['Dimer.1'] == 'Monomer']
print(f"After removing dimers: {len(df)} rows\n")

# Get unique molecules (by InChIKEY)
unique_molecules = df['InChIKEY'].unique()
print(f"Unique molecules: {len(unique_molecules)}")

# Split molecules into train/test (80/20)
train_molecules, test_molecules = train_test_split(unique_molecules, test_size=0.2, random_state=42)

print(f"Train molecules: {len(train_molecules)}")
print(f"Test molecules: {len(test_molecules)}")

# Build datasets based on molecules
train_df = df[df['InChIKEY'].isin(train_molecules)]
test_df = df[df['InChIKEY'].isin(test_molecules)]

print(f"\nTrain rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")

# Verify no shared molecules
shared = set(train_df['InChIKEY'].unique()) & set(test_df['InChIKEY'].unique())
if shared:
    print(f"ERROR: {len(shared)} shared molecules!")
else:
    print("No shared molecules between train and test")

# Save
Path("../data/processed").mkdir(parents=True, exist_ok=True)
train_df.to_csv("../data/processed/train.csv", index=False)
test_df.to_csv("../data/processed/test.csv", index=False)

print(f"\nSaved to:")
print(f" - data/processed/train.csv")
print(f" - data/processed/test.csv")
