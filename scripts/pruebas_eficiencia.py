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

def seleccionar_ejemplos(datos, smiles_input, mz_input, adduct_input, n=100):
    caract_input = extraer_caracteristicas(smiles_input)
    adduct_norm = normalizar_aducto(adduct_input)

    # Separar por aducto
    datos_mismo_aducto = [d for d in datos if normalizar_aducto(d['adduct']) == adduct_norm]
    datos_otros_aductos = [d for d in datos if normalizar_aducto(d['adduct']) != adduct_norm]

    def calcular_similitud(d):
        caract = extraer_caracteristicas(d['smiles'])
        sim_mz = 1/(1 + abs(d['mz'] - mz_input) / 100)
        sim_longitud = 1/(1 + abs(caract['length'] - caract_input['length']) / 10)
        sim_estructura = 1/(1 + abs(caract['num_rings'] - caract_input['num_rings']))
        return 0.35 * sim_mz + 0.15 * sim_longitud + 0.50 * sim_estructura

    # Ordenar ambos grupos por similitud
    mismo_aducto_ordenado = sorted(datos_mismo_aducto, key=calcular_similitud, reverse=True)

    if len(mismo_aducto_ordenado) >= n:
        return mismo_aducto_ordenado[:n]

    # Si no hay suficientes, rellenar con los mejores de otros aductos
    print(f"Aviso: solo {len(mismo_aducto_ordenado)} ejemplos para aducto {adduct_norm}, "
          f"rellenando con {n - len(mismo_aducto_ordenado)} ejemplos de otros aductos")

    otros_ordenados = sorted(datos_otros_aductos, key=calcular_similitud, reverse=True)
    relleno = otros_ordenados[:n - len(mismo_aducto_ordenado)]

    return mismo_aducto_ordenado + relleno


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


def construir_prompt(smiles, mz, adduct, stats, ejemplos, n=10):
    feat = extraer_caracteristicas(smiles)

    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {
        'charge': 1,
        'mass_add': 0,
        'effect': 'unknown adduct type'
    })

    # Recopilar valores prohibidos
    ref_ccs_values = [ej['ccs'] for ej in ejemplos[:n]]
    forbidden = ref_ccs_values + [round(stats['ccs_avg'], 1)]
    forbidden_str = ", ".join(str(v) for v in forbidden)

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
    ccs_low  = max(stats['ccs_min'], ccs_heuristic * 0.85)
    ccs_high = min(stats['ccs_max'], ccs_heuristic * 1.15)

    # Contexto experto adaptado a la molécula concreta
    heteroatom_note = ""
    if feat['num_S'] > 0:
        heteroatom_note += "S adds ~3–5 Å² vs O. "
    if feat['num_N'] > 2:
        heteroatom_note += "Multiple N increases polarity but not size significantly. "
    if feat['aromatic_atoms'] > 12:
        heteroatom_note += "High aromatic count → compact planar shape, CCS lower than aliphatic equivalent. "

    adduct_note = {
        '[M+H]+': "Protonated: standard CCS, no adduct size penalty.",
        '[M+Na]+': "Na adduct: +2–4 Å² vs [M+H]+ due to larger ion radius.",
        '[M-H]-': "Deprotonated: typically –2–5 Å² vs [M+H]+ (tighter ion shape).",
        '[M+K]+': "K adduct: +4–6 Å² vs [M+H]+.",
    }.get(adduct_norm, "")

    prompt = f"""Predict the CCS (Å²) of this molecule.

    EXPERT KNOWLEDGE (apply when adjusting):
    - Each extra ring: +3–6 Å² | Each extra branch: +2–5 Å²
    - Rigid/planar rings (aromatic) expand less than flexible rings
    - Heteroatoms: {heteroatom_note if heteroatom_note else "No special heteroatom effects."}
    - Adduct effect: {adduct_note}

    TARGET: rings={feat['num_rings']} | branches={feat['num_branches']} | aromatic={feat['aromatic_atoms']} | m/z={mz:.4f} | adduct={adduct_norm}

    REFERENCES (Δ = ref minus target):
    {ejemplos_texto}
    VALID RANGE: {ccs_low:.1f}–{ccs_high:.1f} Å²
    FORBIDDEN: {forbidden_str}

    Reasoning (be brief):
    - Best reference: [CCS of ref with Δrings=0 and smallest |Δm/z|]
    - Branch/ring adjustment: [+/- N Å²] because [one reason]
    - Adduct/heteroatom correction: [+/- N Å² or "none"]
    - Final estimate: [base] [+/-] [total] = [RESULT]

    Answer (number only):
    """
    return prompt


def parsear_respuesta(respuesta_raw):
    """
    Extrae el número CCS de la respuesta del modelo.
    Estrategia 1: número inmediatamente después de </reasoning>
    Estrategia 2: último float de 3 dígitos en el texto
    Estrategia 3: cualquier número de 3 dígitos
    """

    # Limpiar artefactos markdown
    texto = respuesta_raw.replace('**', '').replace('*', '').strip()

    # Estrategia 1: número justo después de </reasoning>
    match = re.search(r'</reasoning>\s*([0-9]{2,3}\.?[0-9]*)', texto)
    if match:
        valor = float(match.group(1))
        if 100 <= valor <= 400:  # rango fisiológico de CCS
            return {'predicted_ccs': round(valor, 2), 'fallback': False}

    # Estrategia 2: último float con 3 dígitos antes del punto decimal
    matches = re.findall(r'\b([1-2][0-9]{2}\.[0-9]+)\b', texto)
    if matches:
        return {'predicted_ccs': round(float(matches[-1]), 2), 'fallback': False}

    # Estrategia 3: último entero de 3 dígitos en rango CCS
    matches = re.findall(r'\b([1-2][0-9]{2})\b', texto)
    if matches:
        return {'predicted_ccs': round(float(matches[-1]), 2), 'fallback': False}

    # Sin resultado válido
    return {'predicted_ccs': None, 'fallback': True}


def clasificar_prediccion(ccs_pred, ejemplos, stats):
    """
    Clasifica si la predicción es válida o es un valor degenerado.
    Devuelve (tipo, es_valida).
    """
    ccs_referencias = {round(ej['ccs'], 2) for ej in ejemplos}
    avg = round(stats['ccs_avg'], 1)

    if abs(ccs_pred - avg) <= 0.1:
        return 'dataset_avg', False

    if any(abs(ccs_pred - ref) <= 0.02 for ref in ccs_referencias):
        return 'exact_copy', False

    return 'interpolated', True


def predecir_ccs(model, tokenizer, prompt, mz_fallback, stats, ejemplos):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=5000)
    print(f" Tokens del prompt: {inputs['input_ids'].shape[1]}")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample = True,           # 1.5B -> False ; 7B/14B -> True # True para el 7B
            temperature = 0.3,           # 1.5B -> None  ; 7B/14B -> 0.3 # Bajo para mantener precisión
            top_p = 0.9,                 # 1.5B -> None  ; 7B/14B -> 0.9
            repetition_penalty = 1.15,   # 1.5B -> None  ; 7B/14B -> 1.15 # Penaliza repeticiones
            max_new_tokens = 1000, # más margen para el bloque <reasoning>
            pad_token_id = tokenizer.eos_token_id
            # use_cache = False # Evita corrupción del cache
        )

    respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_texto = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    respuesta = respuesta_completa[len(prompt_texto):].strip()

    print(" RESPUESTA COMPLETA: " + json.dumps(respuesta_completa))
    print(" RESPUESTA CRUDA: " + json.dumps(respuesta))

    resultado = parsear_respuesta(respuesta)

    # --- Fallback por fallo de parseo ---
    if resultado['fallback']:
        ratio = (mz_fallback - stats['mz_min']) / max(stats['mz_max'] - stats['mz_min'], 1)
        resultado['predicted_ccs'] = round(
            stats['ccs_min'] + ratio * (stats['ccs_max'] - stats['ccs_min']), 2
        )
        resultado['pred_type'] = 'heuristic_fallback'
        resultado['reasoning'] = "Heuristic fallback: parser found no valid number"
        return resultado

    # --- Clasificar la predicción ---
    tipo, es_valida = clasificar_prediccion(resultado['predicted_ccs'], ejemplos, stats)
    resultado['predicted_ccs_raw'] = resultado['predicted_ccs']
    resultado['pred_type'] = tipo

    if es_valida:
        resultado['reasoning'] = f"Model interpolation ({resultado['predicted_ccs']})"
    else:
        resultado['pred_type'] = tipo  # 'exact_copy' o 'dataset_avg', sin →heuristic

        if tipo == 'exact_copy':
            resultado['reasoning'] = f"Model output accepted (reference copy: {resultado['predicted_ccs_raw']})"
        else:
            resultado['reasoning'] = f"Model output accepted (dataset avg: {resultado['predicted_ccs_raw']})"
        # predicted_ccs queda tal cual, sin reemplazar

    return resultado


def cargar_modelo():

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model_path = r"D:\Modelos TFG\DeepSeek-R1-Distill-Qwen-1.5B"  # Ruta local
    # model_path = r"/data/guillermo/LuisModelos/DeepSeek-R1-Distill-Qwen-7B"  # Ruta local
    print(f" - Ruta: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = "cpu",
        trust_remote_code = True,
        dtype = torch.float16,
        low_cpu_mem_usage = True
    )

    model.eval()  # Modo inferencia: desactiva dropout y gradientes

    if hasattr(torch, 'compile'):
        print(" - Compilando modelo con torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
        print(" - Compilación completada")

    return model, tokenizer


def inicializar_app():
    global MODEL, TOKENIZER, DATOS_TRAIN, STATS

    print("=" * 70)
    print("Inicializando aplicación de predicción CCS.")
    print("=" * 70)

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
        {"smiles": "COc1cccc(c1)[C@@H]1N(Cc2ccc(cc2)F)C(=O)c2c([C@@H]1C(=O)O)cccc2", "mz": 406.1449, "adduct": "[M+H]+"},
        {"smiles": "OC(=O)/C=C/c1ccc(cc1)OC(F)(F)F", "mz": 231.0275, "adduct": "[M-H]-"},
        {"smiles": "O=C1N[C@@H]2[C@H](N1)[C@@H](SC2)CCCCC(=O)N1CCC(CC1)C(=O)Nc1ccc2c(c1)OCO2", "mz": 473.1864, "adduct": "[M-H]-"},
        {"smiles": "Clc1ccc(cc1)c1occ(n1)CSc1nnnn1CCc1cccs1", "mz": 426.022,  "adduct": "[M+Na]+"},
        {"smiles": "N#CC1(CCCC1)NC(=O)CSc1ccc(cn1)S(=O)(=O)N1CCCCC1", "mz": 431.1182, "adduct": "[M+Na]+"},
        {"smiles": "Oc1cccc(c1)I", "mz": 218.9312, "adduct": "[M-H]"},
        {"smiles": "CCOc1cc2CC(Oc2cc1NC(=O)CN1C(=O)CCOc2c1cccc2)C", "mz": 397.1758, "adduct": "[M+H]"},
        {"smiles": "Cc1ccc(cc1)n1nnnc1SCC(=O)N1CCCC1", "mz": 326.1046, "adduct": "[M+Na]"},
        {"smiles": "COc1ccc2c(c1)CCCC12NC(=O)N(C1=O)CN1CCN(CC1)c1ccccc1", "mz": 419.2089, "adduct": "[M-H]"}
    ]

    print("=" * 70)
    print("PRUEBA DE EFICIENCIA")

    for i, caso in enumerate(test_cases, 1):
        smiles, mz, adduct = caso["smiles"], caso["mz"], caso["adduct"]
        print(f"{'='*70}")
        print(f"COMPUESTO {i} | m/z = {mz} | Aducto = {adduct}")
        print(f"SMILES: {smiles}")
        print(f"{'='*70}")

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

    print(f"\n{'='*70}")
    print("TEST COMPLETADO")
    print("=" * 70)

if __name__ == '__main__':
    if inicializar_app():
        test_prompt()  # <-- cambia esto por app.run(...) cuando quieras volver al servidor
        # app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        print("ERROR en la inicialización. Verifica la configuración.")
