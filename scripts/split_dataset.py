import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Cargar dataset
df = pd.read_csv("data/raw/dataset.csv")

print(f"Dataset original: {len(df)} filas")

# Eliminar filas con valores nulos en columnas clave
df = df.dropna(subset=['smiles', 'InChIKEY', 'CCS_AVG', 'Adduct'])
print(f"Después de limpiar: {len(df)} filas\n")

# Obtener moléculas únicas (por InChIKEY)
unique_molecules = df['InChIKEY'].unique()
print(f"Moléculas únicas: {len(unique_molecules)}")

# Dividir moléculas en train/test (80/20)
train_molecules, test_molecules = train_test_split(unique_molecules, test_size=0.2, random_state=42)

print(f"Moléculas train: {len(train_molecules)}")
print(f"Moléculas test: {len(test_molecules)}")

# Crear datasets basados en las moléculas
train_df = df[df['InChIKEY'].isin(train_molecules)]
test_df = df[df['InChIKEY'].isin(test_molecules)]

print(f"\nFilas train: {len(train_df)}")
print(f"Filas test: {len(test_df)}")

# Verificar que no hay moléculas compartidas
shared = set(train_df['InChIKEY'].unique()) & set(test_df['InChIKEY'].unique())
if shared:
    print(f"ERROR: {len(shared)} moléculas compartidas!")
else:
    print("No hay moléculas compartidas entre train y test")

# Guardar
Path("data/processed").mkdir(parents=True, exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print(f"\nGuardado en:")
print(f" -data/processed/train.csv")
print(f" -data/processed/test.csv")
