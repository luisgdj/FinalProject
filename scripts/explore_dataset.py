import pandas as pd
from pathlib import Path

# Cargar dataset
dataset_path = Path("../data/raw")
csv_file = list(dataset_path.glob("*.csv"))[0]
df = pd.read_csv(csv_file)

print(f"\nDataset: {csv_file.name}")
print(f" - Filas: {len(df)}")
print(f" - Columnas: {len(df.columns)}\n")

# Mostrar columnas
print("Columnas disponibles:")
print(df.columns.tolist())

# Primeras filas
print("\nPrimeras filas:")
print(df.head(3))

# Estadísticas básicas
print("\nInformación:")
print(df.info())
