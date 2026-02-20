import pandas as pd
import sys
from pathlib import Path

# Añadir directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.model_loader import model_loader
from backend.agent import agent, CompoundInput

"""
EXPERIMENTOS DE PREDICCIÓN CCS
- Experimento A: Zero-shot (sin ejemplos)
- Experimento B: Few-shot (con ejemplos del training)
"""

print("=" * 70)
print("EXPERIMENTOS DE PREDICCIÓN CCS")
print("=" * 70)

# Cargar modelo
print("\n⏳ Cargando modelo DeepSeek...")
model_loader.cargar_modelo()

# Cargar datasets
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print(f"\n Train: {len(train_df)} filas")
print(f" Test: {len(test_df)} filas")

# Tomar muestra del test (20 compuestos aleatorios)
# sample = test_df.sample(n=20, random_state=42)
sample = train_df.sample(n=20, random_state=42)
print(f"\nUsando muestra de {len(sample)} compuestos del test")

# Preparar ejemplos para few-shot
print("\nPreparando ejemplos para few-shot...")
ejemplos = []
train_sample = train_df.sample(n=5, random_state=42)

for _, row in train_sample.iterrows():
    ejemplos.append({
        'smiles': row['smiles'],
        'adduct': row['Adduct'],
        'ccs': row['CCS_AVG']
    })

print(f" {len(ejemplos)} ejemplos preparados")

# ========================================
# EXPERIMENTO A: ZERO-SHOT
# ========================================
print("\n" + "=" * 70)
print("EXPERIMENTO A: ZERO-SHOT (sin ejemplos)")
print("=" * 70)

resultados_zero = []

for i, (idx, row) in enumerate(sample.iterrows(), 1):
    print(f"Prediciendo {i}/{len(sample)}...", end='\r')

    compound = CompoundInput(
        smiles=row['smiles'],
        adduct=row['Adduct'],
        is_dimer=bool(row['Dimer']),
        mz=float(row['m/z']) if pd.notna(row['m/z']) else None
    )

    # Predecir SIN ejemplos
    resultado = agent.predecir(compound, model_loader, ejemplos=None)

    resultados_zero.append({
        'smiles': row['smiles'],
        'ccs_real': row['CCS_AVG'],
        'ccs_predicho': resultado['prediction']['ccs'],
        'error': abs(row['CCS_AVG'] - resultado['prediction']['ccs'])
    })

results_zero_df = pd.DataFrame(resultados_zero)
mae_zero = results_zero_df['error'].mean()

print(f"\nZero-shot completado")
print(f"   MAE: {mae_zero:.2f} Ų")

# ========================================
# EXPERIMENTO B: FEW-SHOT
# ========================================
print("\n" + "=" * 70)
print("EXPERIMENTO B: FEW-SHOT (con 5 ejemplos)")
print("=" * 70)

resultados_few = []

for i, (idx, row) in enumerate(sample.iterrows(), 1):
    print(f"Prediciendo {i}/{len(sample)}...", end='\r')

    compound = CompoundInput(
        smiles=row['smiles'],
        adduct=row['Adduct'],
        is_dimer=bool(row['Dimer']),
        mz=float(row['m/z']) if pd.notna(row['m/z']) else None
    )

    # Predecir CON ejemplos
    resultado = agent.predecir(compound, model_loader, ejemplos=ejemplos)

    resultados_few.append({
        'smiles': row['smiles'],
        'ccs_real': row['CCS_AVG'],
        'ccs_predicho': resultado['prediction']['ccs'],
        'error': abs(row['CCS_AVG'] - resultado['prediction']['ccs'])
    })

results_few_df = pd.DataFrame(resultados_few)
mae_few = results_few_df['error'].mean()

print(f"\nFew-shot completado")
print(f"   MAE: {mae_few:.2f} Ų")

# ========================================
# RESULTADOS Y COMPARACIÓN
# ========================================
print("\n" + "=" * 70)
print("RESULTADOS FINALES")
print("=" * 70)

print(f"\n MAE Zero-shot:  {mae_zero:.2f} Ų")
print(f" MAE Few-shot:   {mae_few:.2f} Ų")

mejora = mae_zero - mae_few
porcentaje = (mejora / mae_zero) * 100 if mae_zero > 0 else 0

if mejora > 0:
    print(f"\nMejora con few-shot: {mejora:.2f} Ų ({porcentaje:.1f}%)")
else:
    print(f"\n⚠Few-shot no mejoró: {mejora:.2f} Ų ({porcentaje:.1f}%)")

# Mostrar algunos resultados
print("\n" + "=" * 70)
print("PRIMEROS 5 RESULTADOS")
print("=" * 70)
print("\nZero-shot:")
print(results_zero_df[['ccs_real', 'ccs_predicho', 'error']].head())
print("\nFew-shot:")
print(results_few_df[['ccs_real', 'ccs_predicho', 'error']].head())

# Guardar resultados
results_zero_df.to_csv("data/processed/results_zeroshot.csv", index=False)
results_few_df.to_csv("data/processed/results_fewshot.csv", index=False)

print("\n" + "=" * 70)
print("EXPERIMENTOS COMPLETADOS")
print("=" * 70)
print("\nResultados guardados en:")
print(" - data/processed/results_zeroshot.csv")
print(" - data/processed/results_fewshot.csv")
print("\n" + "=" * 70)
