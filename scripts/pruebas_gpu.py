from flask import Flask
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import csv

app = Flask(__name__)

MODEL = None
TOKENIZER = None
DATOS_TRAIN = None
STATS = None


def extraer_caracteristicas(smiles):
    return {
        "length": len(smiles),
        "num_rings": sum(c.isdigit() for c in smiles),
        "num_branches": smiles.count("("),
        "double_bonds": smiles.count("="),
        "triple_bonds": smiles.count("#"),
        "aromatic_atoms": sum(smiles.count(c) for c in "cnosp"),
        "num_N": smiles.count("N") + smiles.count("n"),
        "num_O": smiles.count("O") + smiles.count("o"),
        "num_S": smiles.count("S") + smiles.count("s"),
        "num_P": smiles.count("P"),
        "num_F": smiles.count('F'),
        "num_Cl": smiles.count('Cl'),
        "num_Br": smiles.count('Br'),
        "stereocenters": smiles.count("@"),
        "net_charge": smiles.count("+") - smiles.count("-"),
    }


def calcular_correlacion(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / ((den_x * den_y) ** 0.5)


def calcular_std(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return variance ** 0.5


def leer_csv(filepath):
    datos = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                datos.append({
                    'smiles': row['smiles'].strip(),
                    'adduct': row['Adduct'].strip(),
                    'mz': float(row['m/z']),
                    'ccs': float(row['CCS_AVG'])
                })
            except (ValueError, KeyError):
                continue
    return datos


def analizar_datos(datos):
    ccs_values = [d['ccs'] for d in datos]
    mz_values = [d['mz'] for d in datos]
    corr = calcular_correlacion(mz_values, ccs_values)
    stats = {
        'ccs_min': min(ccs_values),
        'ccs_max': max(ccs_values),
        'ccs_avg': sum(ccs_values) / len(ccs_values),
        'ccs_std': calcular_std(ccs_values),
        'correlacion_mz_ccs': corr,
        'mz_min': min(mz_values),
        'mz_max': max(mz_values),
    }
    return stats


def normalizar_aducto(adduct):
    # Elimina la carga final del aducto para unificar formatos.
    return adduct.rstrip('+-').strip()


def seleccionar_ejemplos(datos, smiles_input, mz_input, adduct_input, n=10):
    caract_input = extraer_caracteristicas(smiles_input)
    adduct_norm = normalizar_aducto(adduct_input)

    # Filtrar solo compuestos con el mismo aducto
    datos_filtrados = [d for d in datos if normalizar_aducto(d['adduct']) == adduct_norm]

    # Si no hay suficientes ejemplos del mismo aducto, usar todos
    if len(datos_filtrados) < n:
        print(f"Aviso: solo {len(datos_filtrados)} ejemplos para aducto {adduct_norm}, usando dataset completo")
        datos_filtrados = datos

    similitudes = []
    for d in datos_filtrados:
        caract = extraer_caracteristicas(d['smiles'])

        # La fórmula 1/(1 + diferencia) convierte distancias en similitudes en rango (0, 1]
        sim_mz = 1 / (1 + abs(d['mz'] - mz_input) / 100)  # Mide la proximidad en masas, tiene un peso del 35%
        sim_longitud = 1 / (1 + abs(
            caract['length'] - caract_input['length']) / 10)  # Mide la longitud del SMILES, tiene un peso del 15%
        sim_estructura = 1 / (1 + abs(
            caract['num_rings'] - caract_input['num_rings']))  # Mide el número de anillos, tiene un peso del 50%

        # Podría usar mas caracteristicas del SMILES pero este proceso orientativo y no es necesario
        similitud = 0.35 * sim_mz + 0.15 * sim_longitud + 0.50 * sim_estructura
        similitudes.append((similitud, d))

    similitudes.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in similitudes]


def buscar_en_dataset(smiles, adduct, dataset):
    adduct_norm = normalizar_aducto(adduct)
    for row in dataset:
        if row["smiles"] == smiles and normalizar_aducto(row["Adduct"]) == adduct_norm:
            return row["ccs"]
    return None


ADDUCT_INFO = {
    '[M+H]': {'charge': 1, 'mass_add': 1.007, 'effect': 'standard reference, protonated'},
    '[M+Na]': {'charge': 1, 'mass_add': 22.989, 'effect': 'sodium adduct, slightly larger CCS than [M+H]+'},
    '[M+K]': {'charge': 1, 'mass_add': 38.963, 'effect': 'potassium adduct, larger CCS than [M+Na]+'},
    '[M-H]': {'charge': -1, 'mass_add': -1.007, 'effect': 'deprotonated negative mode, typically smaller CCS'},
    '[M+NH4]': {'charge': 1, 'mass_add': 18.034, 'effect': 'ammonium adduct, bulkier than [M+H]+'},
    '[M+2H]2': {'charge': 2, 'mass_add': 2.014, 'effect': 'doubly charged, molecule compacts, lower CCS per charge'},
    '[M+FA-H]': {'charge': -1, 'mass_add': 44.998, 'effect': 'formate adduct negative mode'},
    '[M+Hac-H]': {'charge': -1, 'mass_add': 59.013, 'effect': 'acetate adduct negative mode'},
}


def construir_prompt(smiles, mz, adduct, stats, ejemplos, n=5):
    feat = extraer_caracteristicas(smiles)

    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {
        'charge': 1,
        'mass_add': 0,
        'effect': 'unknown adduct type'
    })

    ejemplos_texto = ""
    for ej in ejemplos[:n]:
        feat_ej = extraer_caracteristicas(ej['smiles'])
        ejemplos_texto += (
            f"  SMILES={ej['smiles']} | adduct={ej['adduct']} | "
            f"m/z={ej['mz']} | rings={feat_ej['num_rings']} | "
            f"branches={feat_ej['num_branches']} | CCS={ej['ccs']}\n"
        )

    # Ancla heurística basada en m/z (evita que el modelo devuelva el promedio)
    ratio = (mz - stats['mz_min']) / max(stats['mz_max'] - stats['mz_min'], 1)
    ccs_heuristic = stats['ccs_min'] + ratio * (stats['ccs_max'] - stats['ccs_min'])

    # Rango esperado: ±15% alrededor de la heurística, acotado al dataset
    ccs_low = max(stats['ccs_min'], ccs_heuristic * 0.85)
    ccs_high = min(stats['ccs_max'], ccs_heuristic * 1.15)

    prompt = f"""You are a mass spectrometry expert specializing in ion mobility. Estimate the CCS (Å²) of the target molecule.

CCS measures the rotationally-averaged collision cross-section: larger, more branched, or more rigid molecules have higher CCS.

TARGET
------
SMILES: {smiles}
m/z: {mz:.4f} | Adduct: {adduct_norm} ({info['effect']}, charge {info['charge']:+d})
Rings: {feat['num_rings']} | Branches: {feat['num_branches']} | Aromatic atoms: {feat['aromatic_atoms']}
N:{feat['num_N']} O:{feat['num_O']} S:{feat['num_S']} | Net charge: {feat['net_charge']}

REFERENCE MOLECULES (ranked by structural similarity — use for interpolation, do not copy values)
-------------------------------------------------------------------------------------------------
{ejemplos_texto}
DATASET STATISTICS
------------------
CCS range: {stats['ccs_min']:.1f}–{stats['ccs_max']:.1f} Å² | Average: {stats['ccs_avg']:.1f} Å²
m/z-based heuristic estimate: {ccs_heuristic:.1f} Å² (expected range: {ccs_low:.1f}–{ccs_high:.1f} Å²)
NOTE: the heuristic is a lower bound — use references for the final estimate.

RULES
-----
1. Identify the 2–3 most similar references (same adduct, similar rings and branches).
2. Interpolate: more rings/branches → higher CCS; smaller molecule → lower CCS.
3. Your answer MUST be a single number different from all reference CCS values.
4. If the target is smaller than references, predict below their range.
5. Never return the dataset average ({stats['ccs_avg']:.1f}) as your answer.

Think step by step before answering.

OUTPUT (one number only, no extra text):
Based on the structure and the references, the predicted CCS is approximately """
    return prompt


def parsear_respuesta(respuesta):
    resultado = {}

    # Priorizar texto después de </think>
    if "</think>" in respuesta:
        texto = respuesta.split("</think>")[-1].strip()
    elif "<think>" in respuesta:
        texto = respuesta.split("<think>")[0].strip()
    else:
        texto = respuesta.strip()

    # Captura números con o sin markdown bold (**201.58** o 201.58 o 195.9)
    numeros = re.findall(r'\*{0,2}(\d{2,3}(?:\.\d+)?)\*{0,2}', texto)
    for num_str in numeros:
        ccs = float(num_str)
        if 130 < ccs < 280:
            resultado['predicted_ccs'] = round(ccs, 2)
            break

    if 'predicted_ccs' not in resultado:
        numeros = re.findall(r'\*{0,2}(\d{2,3}(?:\.\d+)?)\*{0,2}', respuesta)
        for num_str in numeros:
            ccs = float(num_str)
            if 130 < ccs < 280:
                resultado['predicted_ccs'] = round(ccs, 2)
                break

    resultado['confidence'] = 'unknown'
    resultado['fallback'] = 'predicted_ccs' not in resultado
    return resultado


def clasificar_prediccion(ccs_pred, ejemplos, stats):
    # Lista ordenada de CCS de referencia
    ccs_referencias = {ej['ccs'] for ej in ejemplos}
    avg = stats['ccs_avg']

    # CASO 1: Comprobar si es la media del dataset
    if abs(ccs_pred - avg) <= 0.1:
        return 'dataset_avg', ccs_pred
    # CASO 2: Comprobar si es copia exacta de referencia
    if any(abs(ccs_pred - ref) <= 0.02 for ref in ccs_referencias):
        return 'exact_copy', ccs_pred
    # CASO 3: Interpolación propia del modelo
    return 'interpolated', ccs_pred


def predecir_ccs(model, tokenizer, prompt, mz_fallback, stats, ejemplos):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    # Sin .cuda() manual — device_map="auto" gestiona los dispositivos
    print(f" Tokens del prompt: {inputs['input_ids'].shape[1]}")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=80,
            pad_token_id=tokenizer.eos_token_id
        )

    respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_texto = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    respuesta = respuesta_completa[len(prompt_texto):].strip()

    print(" RESPUESTA COMPLETA: " + json.dumps(respuesta_completa))
    print(" RESPUESTA CRUDA: " + json.dumps(respuesta))

    resultado = parsear_respuesta(respuesta)

    if resultado['fallback']:
        ratio = (mz_fallback - stats['mz_min']) / max(stats['mz_max'] - stats['mz_min'], 1)
        resultado['predicted_ccs'] = round(
            stats['ccs_min'] + ratio * (stats['ccs_max'] - stats['ccs_min']), 2
        )
        resultado['pred_type'] = 'heuristic_fallback'
        resultado['reasoning'] = "Heuristic fallback based on m/z interpolation"
    else:
        tipo, ccs_final = clasificar_prediccion(resultado['predicted_ccs'], ejemplos, stats)
        resultado['pred_type'] = tipo
        resultado['predicted_ccs_raw'] = resultado['predicted_ccs']
        resultado['predicted_ccs'] = ccs_final

        if tipo == 'exact_copy':
            resultado['reasoning'] = f"Model copied reference value ({ccs_final})"
        elif tipo == 'dataset_avg':
            resultado['reasoning'] = f"Model returned dataset average ({ccs_final})"
        else:
            resultado['reasoning'] = f"Model interpolation ({ccs_final})"

    return resultado


def cargar_modelo():
    model_path = r"D:\Modelos TFG\DeepSeek-R1-Distill-Qwen-1.5B"  # 1.5B, el 7B no cabe en 2GB VRAM
    print(f" - Ruta: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="gpu", # auto - Distribuye entre GPU y CPU automáticamente
        trust_remote_code=True,
        torch_dtype=torch.float16, # float16 obligatorio en GPU
        low_cpu_mem_usage=True
    )

    model.eval()
    return model, tokenizer



def inicializar_app():
    global MODEL, TOKENIZER, DATOS_TRAIN, STATS

    print("=" * 70)
    print("Inicializando aplicación de predicción CCS.")
    print("=" * 70)

    # Verificar GPU
    if torch.cuda.is_available():
        print(f" - GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f" - VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print(" - ADVERTENCIA: CUDA no disponible, usando CPU")

    # Cargar datos
    csv_path = r"../data/processed/train.csv"
    if not os.path.exists(csv_path):
        print(f"AVISO: Archivo {csv_path} no encontrado")
        print("Por favor, asegúrate de que train.csv está en la ruta correcta")
        return False

    print(f"Cargando dataset: {csv_path}")
    DATOS_TRAIN = leer_csv(csv_path)
    print(f" - Dataset cargado: {len(DATOS_TRAIN)} compuestos")

    # Analizar estadísticas
    print("Analizando estadísticas del dataset...")
    STATS = analizar_datos(DATOS_TRAIN)
    print(f" - CCS range: {STATS['ccs_min']:.1f} - {STATS['ccs_max']:.1f} Å²")
    print(f" - Correlación m/z-CCS: {STATS['correlacion_mz_ccs']:.3f}")

    # Cargar modelo
    print("Cargando modelo DeepSeek...")
    MODEL, TOKENIZER = cargar_modelo()
    print(" - Modelo cargado y listo")
    print(" - Aplicación lista para recibir peticiones")
    print("=" * 70)
    return True


def test_prompt():
    global MODEL, TOKENIZER, DATOS_TRAIN, STATS

    # Ejemplos de la tabla
    test_cases = [
        {"smiles": "O=C([C@@H](NS(=O)(=O)c1ccc(cc1)Cl)Cc1c[nH]c2c1cccc2)NC1CCCC1", "mz": 446.13, "adduct": "[M+H]+"},
        {"smiles": "COc1cccc(c1)[C@@H]1N(Cc2ccc(cc2)F)C(=O)c2c([C@@H]1C(=O)O)cccc2", "mz": 406.1449,
         "adduct": "[M+H]+"},
        {"smiles": "OC(=O)/C=C/c1ccc(cc1)OC(F)(F)F", "mz": 231.0275, "adduct": "[M-H]-"},
        {"smiles": "O=C1N[C@@H]2[C@H](N1)[C@@H](SC2)CCCCC(=O)N1CCC(CC1)C(=O)Nc1ccc2c(c1)OCO2", "mz": 473.1864,
         "adduct": "[M-H]-"},
        {"smiles": "Clc1ccc(cc1)c1occ(n1)CSc1nnnn1CCc1cccs1", "mz": 426.022, "adduct": "[M+Na]+"},
        {"smiles": "N#CC1(CCCC1)NC(=O)CSc1ccc(cn1)S(=O)(=O)N1CCCCC1", "mz": 431.1182, "adduct": "[M+Na]+"},
    ]

    print("=" * 70)
    print("PRUEBA DE EFICIENCIA")

    for i, caso in enumerate(test_cases, 1):
        smiles, mz, adduct = caso["smiles"], caso["mz"], caso["adduct"]
        print(f"{'=' * 70}")
        print(f"COMPUESTO {i} | m/z = {mz} | Aducto = {adduct}")
        print(f"SMILES: {smiles}")
        print(f"{'=' * 70}")

        # Limpiar caché entre predicciones
        torch.cuda.empty_cache()  # por si acaso aunque estés en CPU
        if hasattr(MODEL, 'reset_cache'):
            MODEL.reset_cache()

        ejemplos = seleccionar_ejemplos(DATOS_TRAIN, smiles, mz, adduct)
        prompt = construir_prompt(smiles, mz, adduct, STATS, ejemplos)
        resultado = predecir_ccs(MODEL, TOKENIZER, prompt, mz, STATS, ejemplos)

        fallback_str = " [FALLBACK]" if resultado["fallback"] else ""
        print(f" CCS = {resultado['predicted_ccs']:.2f} Ų{fallback_str}")
        print(f" Reasoning: {resultado['reasoning'][:80]}")

    print(f"\n{'=' * 70}")
    print("TEST COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    if inicializar_app():
        test_prompt()  # <-- cambia esto por app.run(...) cuando quieras volver al servidor
        # app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        print("ERROR en la inicialización. Verifica la configuración.")
