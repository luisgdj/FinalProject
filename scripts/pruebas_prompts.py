import sys
from datetime import datetime

from flask import Flask
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import csv
import re

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

# Selecciona N moléculas del dataset estructuralmente similares al input.
def seleccionar_ejemplos(datos, smiles_input, adduct_input, n):

    caract_input = extraer_caracteristicas(smiles_input)
    adduct_norm = normalizar_aducto(adduct_input)

    # Filtrar por mismo aducto (los CCS dependen del aducto)
    datos_filtrados = [d for d in datos if normalizar_aducto(d['adduct']) == adduct_norm]
    if len(datos_filtrados) < n:
        print(f"Aviso: solo {len(datos_filtrados)} ejemplos para aducto {adduct_norm}, usando dataset completo")
        datos_filtrados = datos

    similitudes = []
    for d in datos_filtrados:
        caract = extraer_caracteristicas(d['smiles'])

        # Tres criterios de similitud estructural (todos en rango 0-1)
        sim_longitud   = 1 / (1 + abs(caract['length']    - caract_input['length']) / 10)
        sim_anillos    = 1 / (1 + abs(caract['num_rings'] - caract_input['num_rings']))
        sim_aromatico  = 1 / (1 + abs(caract['aromatic_atoms'] - caract_input['aromatic_atoms']) / 3)

        # Pesos: estructura > longitud > aromaticidad
        similitud = 0.50 * sim_anillos + 0.30 * sim_longitud + 0.20 * sim_aromatico
        similitudes.append((similitud, d))

    similitudes.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in similitudes[:n]]


def buscar_en_dataset(smiles, adduct, dataset):
    adduct_norm = normalizar_aducto(adduct)
    for row in dataset:
        if row["smiles"] == smiles and normalizar_aducto(row["Adduct"]) == adduct_norm:
            return row["ccs"]
    return None


ADDUCT_INFO = {
    '[M+H]':     {'charge': 1,  'mass_add': 1.007,  'effect': 'protonated, baseline reference'},
    '[M+Na]':    {'charge': 1,  'mass_add': 22.989, 'effect': 'sodium adduct, ~3-5 Å² larger than [M+H]+'},
    '[M+K]':     {'charge': 1,  'mass_add': 38.963, 'effect': 'potassium adduct, ~5-8 Å² larger than [M+H]+'},
    '[M-H]':     {'charge': -1, 'mass_add': -1.007, 'effect': 'deprotonated, ~2-5 Å² smaller than [M+H]+'},
    '[M+NH4]':   {'charge': 1,  'mass_add': 18.034, 'effect': 'ammonium adduct, ~5-10 Å² larger than [M+H]+'},
    '[M+2H]2':   {'charge': 2,  'mass_add': 2.014,  'effect': 'doubly charged, ~30-40% smaller CCS (compact)'},
    '[M+FA-H]':  {'charge': -1, 'mass_add': 44.998, 'effect': 'formate adduct (negative mode), ~5-10 Å² larger than [M-H]-'},
    '[M+Hac-H]': {'charge': -1, 'mass_add': 59.013, 'effect': 'acetate adduct (negative mode), ~7-12 Å² larger than [M-H]-'},
}

# Iteración 1: Monta un prompt simple solo usando la información del compuesto a analizar
def construir_prompt_simple(smiles, adduct):
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {
        'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'
    })

    prompt = f"""You are an expert in ion mobility mass spectrometry.
Your task is to predict the Collision Cross Section (CCS, in Å²) of the molecule from its SMILES structure and adduct.

To estimate CCS, follow these steps:
  1. Read the SMILES and identify rings, chains, branches, and functional groups.
  2. Decide if the structure is compact (mostly fused rings, rigid) or extended (chains, flexible).
  3. Apply the adduct correction described in the molecule line below.
  4. Output a single CCS value, typically in the range 100-300 Å² for small molecules.

Molecule:
   SMILES: {smiles}
   Adduct: {adduct_norm} ({info['effect']})

End your answer with the line "Final CCS: <number>".
<think>
Let me apply the 4 steps to estimate the CCS.

Step 1 (size): """
    return prompt

# Iteración 2: Monta un prompt usando conocimiento experto de metabolómica y la información del compuesto a analizar
def construir_prompt_ce(smiles, adduct):
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {
        'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'
    })

    prompt = f"""You are an expert in ion mobility mass spectrometry.
Your task is to predict the Collision Cross Section (CCS, in Å²) of the molecule from its SMILES structure and adduct.

CCS reflects the rotationally-averaged area a molecule presents while colliding with buffer gas in an ion mobility cell. It is determined primarily by the three-dimensional shape of the ion, not by its mass.

Expert knowledge - structural factors that determine CCS:

1. Molecular size and rigidity:
   - Compact, rigid structures (fused aromatic systems, polycyclic cores) yield LOWER CCS for their mass.
   - Extended, flexible structures (long aliphatic chains, single-bond rotations) yield HIGHER CCS for their mass.

2. Branching and substitution:
   - Branched molecules pack more compactly than linear isomers of the same mass → lower CCS.
   - Bulky substituents (long side chains, multiple rings off a central atom) increase CCS.

3. Heteroatoms and functional groups:
   - Heteroatoms (N, O, S) and halogens add mass but contribute relatively little to molecular volume.
   - Polar groups (-OH, -NH2, -COOH) can promote intramolecular interactions, slightly reducing CCS.

4. Adduct effect:
   - The adduct alters charge state and ion geometry.
   - Use the adduct effect described in the molecule line below as a small final adjustment.

To estimate CCS, follow these steps:
  1. Read the SMILES and identify rings, chains, branches, and functional groups.
  2. Decide if the structure is compact (mostly fused rings, rigid) or extended (chains, flexible).
  3. Apply the adduct correction described in the molecule line below.
  4. Output a single CCS value, typically in the range 100-300 Å² for small molecules.

Molecule:
  SMILES: {smiles}
  Adduct: {adduct_norm} ({info['effect']})

End your answer with the line "Final CCS: <number>".
<think>
Let me apply the 4 steps to estimate the CCS.

Step 1 (structure): """
    return prompt

# Iteración 3: Monta un prompt usando ejemplos de una base de datos y la información del compuesto a analizar
def construir_prompt_ejemplos(smiles, adduct, datos_train, n):
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {
        'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'
    })

    # Seleccionar N moléculas estructuralmente similares del dataset
    ejemplos = seleccionar_ejemplos(datos_train, smiles, adduct, n)

    # Construir bloque de ejemplos (mismo formato que el target: SMILES + Adduct + CCS)
    ejemplos_texto = ""
    for i, ej in enumerate(ejemplos, 1):
        ej_adduct_norm = normalizar_aducto(ej['adduct'])
        ejemplos_texto += (
            f"  Example {i}:\n"
            f"    SMILES: {ej['smiles']}\n"
            f"    Adduct: {ej_adduct_norm}\n"
            f"    CCS: {ej['ccs']:.2f} Å²\n\n"
        )

    prompt = f"""You are an expert in ion mobility mass spectrometry.
Your task is to predict the Collision Cross Section (CCS, in Å²) of the molecule from its SMILES structure and adduct.

Below are {n} reference molecules with experimentally measured CCS values, selected for their structural similarity to the target. Use them to identify patterns and infer the CCS of the target molecule.

Reference molecules:

{ejemplos_texto}To estimate CCS, follow these steps:
  1. Read the SMILES and identify rings, chains, branches, and functional groups.
  2. Decide if the structure is compact (mostly fused rings, rigid) or extended (chains, flexible).
  3. Compare with the reference molecules above. Identify the most similar ones and use their CCS values as anchors.
  4. Apply the adduct correction described in the molecule line below.
  5. Output a single CCS value, typically in the range 100-300 Å² for small molecules.

Molecule:
   SMILES: {smiles}
   Adduct: {adduct_norm} ({info['effect']})

End your answer with the line "Final CCS: <number>".
<think>
Let me apply the 5 steps to estimate the CCS.

Step 1 (size): """
    return prompt

# Iteración 4: Monta un prompt usando conocimiento experto, ejemplos de una base de datos y la información del compuesto a analizar
def construir_prompt_completo(smiles, adduct, datos_train, n):
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {
        'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'
    })

    # Seleccionar N moléculas estructuralmente similares del dataset
    ejemplos = seleccionar_ejemplos(datos_train, smiles, adduct, n)

    # Construir bloque de ejemplos (mismo formato que el target: SMILES + Adduct + CCS)
    ejemplos_texto = ""
    for i, ej in enumerate(ejemplos, 1):
        ej_adduct_norm = normalizar_aducto(ej['adduct'])
        ejemplos_texto += (
            f"  Example {i}:\n"
            f"    SMILES: {ej['smiles']}\n"
            f"    Adduct: {ej_adduct_norm}\n"
            f"    CCS: {ej['ccs']:.2f} Å²\n\n"
        )

    prompt = f"""You are an expert in ion mobility mass spectrometry.
Your task is to predict the Collision Cross Section (CCS, in Å²) of the molecule from its SMILES structure and adduct.

CCS reflects the rotationally-averaged area a molecule presents while colliding with buffer gas in an ion mobility cell. It is determined primarily by the three-dimensional shape of the ion, not by its mass.

Expert knowledge - structural factors that determine CCS:

1. Molecular size and rigidity:
   - Compact, rigid structures (fused aromatic systems, polycyclic cores) yield LOWER CCS for their mass.
   - Extended, flexible structures (long aliphatic chains, single-bond rotations) yield HIGHER CCS for their mass.

2. Branching and substitution:
   - Branched molecules pack more compactly than linear isomers of the same mass → lower CCS.
   - Bulky substituents (long side chains, multiple rings off a central atom) increase CCS.

3. Heteroatoms and functional groups:
   - Heteroatoms (N, O, S) and halogens add mass but contribute relatively little to molecular volume.
   - Polar groups (-OH, -NH2, -COOH) can promote intramolecular interactions, slightly reducing CCS.

4. Adduct effect:
   - The adduct alters charge state and ion geometry.
   - Use the adduct effect described in the molecule line below as a small final adjustment.

To estimate CCS, follow these steps:
  1. Read the SMILES and identify rings, chains, branches, and functional groups.
  2. Decide if the structure is compact (mostly fused rings, rigid) or extended (chains, flexible).
  3. Apply the adduct correction described in the molecule line below.
  4. Output a single CCS value, typically in the range 100-300 Å² for small molecules.

Below are {n} reference molecules with experimentally measured CCS values, selected for their structural similarity to the target. Use them to identify patterns and infer the CCS of the target molecule.

Reference molecules:

{ejemplos_texto}To estimate CCS, follow these steps:
  1. Read the SMILES and identify rings, chains, branches, and functional groups.
  2. Decide if the structure is compact (mostly fused rings, rigid) or extended (chains, flexible).
  3. Compare with the reference molecules above. Identify the most similar ones and use their CCS values as anchors.
  4. Apply the adduct correction described in the molecule line below.
  5. Output a single CCS value, typically in the range 100-300 Å² for small molecules.

Molecule:
   SMILES: {smiles}
   Adduct: {adduct_norm} ({info['effect']})

End your answer with the line "Final CCS: <number>".
<think>
Let me apply the 5 steps to estimate the CCS.

Step 1 (size): """
    return prompt


# Extrae el CCS predicho de la respuesta del modelo
def parsear_respuesta(respuesta_raw):

    # Limpieza inicial
    # Quitamos markdown bold/italic que ensucia las regex
    texto = respuesta_raw.replace('**', '').replace('*', '')
    # Quitamos artefactos LaTeX comunes: $...$, $$...$$, \[ \], \( \)
    texto = re.sub(r'\$+', '', texto)
    texto = re.sub(r'\\[\[\]\(\)]', '', texto)
    # Normalizamos espacios
    texto = texto.strip()

    # Comprobación: ¿el razonamiento se cerró?
    if '<think>' in texto and '</think>' not in texto:
        return {'predicted_ccs': None, 'fallback': True, 'source': 'think_unclosed'}
    if '</think>' not in texto:
        return {'predicted_ccs': None, 'fallback': True, 'source': 'no_think_tag'}

    # A partir de aquí trabajamos SOLO con la zona post-think
    texto_post = texto.split('</think>')[-1].strip()

    # ESTRATEGIA 1: Patrón explícito "Final CCS: <número>"
    # Acepta variantes: "Final CCS:", "Final CCS =", "**Final CCS:**" (ya limpiado),
    # con o sin "Å²" detrás, y permite que el número venga dentro de \boxed{}
    patrones_final = [
        # "Final CCS: \boxed{195}" o "Final CCS: 195"
        r'Final\s+CCS\s*[:=]?\s*\\?boxed\{?\s*([0-9]+(?:\.[0-9]+)?)\s*\}?',
        # "Final CCS: 195" o "Final CCS = 195.5"
        r'Final\s+CCS\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
        # "Final CCS 195" (sin separador)
        r'Final\s+CCS\s+([0-9]+(?:\.[0-9]+)?)',
    ]
    for patron in patrones_final:
        match = re.search(patron, texto_post, re.IGNORECASE)
        if match:
            valor = float(match.group(1))
            if 50 <= valor <= 500:  # rango razonable para CCS
                return {
                    'predicted_ccs': round(valor, 2),
                    'fallback': False,
                    'source': 'final_ccs_tag',
                }

    # ESTRATEGIA 2: Patrón \boxed{<número>} (sin "Final CCS:" delante)
    # DeepSeek-R1 a veces solo usa \boxed{195} como respuesta final
    match = re.search(r'\\?boxed\{?\s*([0-9]+(?:\.[0-9]+)?)\s*\}?', texto_post)
    if match:
        valor = float(match.group(1))
        if 50 <= valor <= 500:
            return {
                'predicted_ccs': round(valor, 2),
                'fallback': False,
                'source': 'boxed_tag',
            }

    # ESTRATEGIA 3: Número inmediatamente después de </think>
    # Por si el modelo no usa el formato "Final CCS:" pero da un número limpio
    match = re.match(r'^([0-9]+(?:\.[0-9]+)?)', texto_post)
    if match:
        valor = float(match.group(1))
        if 50 <= valor <= 500:
            return {
                'predicted_ccs': round(valor, 2),
                'fallback': False,
                'source': 'post_think_first',
            }

    # ESTRATEGIA 4: Último número plausible en la zona post-think
    # Como último recurso, el último número en rango CCS razonable
    matches = re.findall(r'\b([0-9]{2,3}(?:\.[0-9]+)?)\b', texto_post)
    plausibles = [float(m) for m in matches if 100 <= float(m) <= 400]
    if plausibles:
        return {
            'predicted_ccs': round(plausibles[-1], 2),
            'fallback': False,
            'source': 'last_plausible',
        }

    # Sin número válido
    return {'predicted_ccs': None, 'fallback': True, 'source': 'no_number_found'}

# Clasifica si la predicción es válida o es un valor degenerado.
def clasificar_prediccion(ccs_pred, ejemplos, stats):

    ccs_referencias = {round(ej['ccs'], 2) for ej in ejemplos}
    avg = round(stats['ccs_avg'])

    if abs(ccs_pred - avg) <= 0.5:
        return 'dataset_avg', False

    if any(abs(ccs_pred - ref) <= 0.02 for ref in ccs_referencias):
        return 'exact_copy', False

    return 'interpolated', True


def predecir_ccs(model, tokenizer, prompt, mz_fallback, stats, ejemplos):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=5000)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample = False,           # 1.5B -> False ; 7B/14B -> True # True para el 7B
            temperature = None,           # 1.5B -> None  ; 7B/14B -> 0.3 # Bajo para mantener precisión
            top_p = None,                 # 1.5B -> None  ; 7B/14B -> 0.9
            repetition_penalty = 1.15,   # 1.5B -> None  ; 7B/14B -> 1.15 # Penaliza repeticiones
            max_new_tokens = 5000, # más margen para el bloque <reasoning>
            pad_token_id = tokenizer.eos_token_id,
            # use_cache = False # False -> Evita corrupción del cache
        )

    n_tokens_prompt = inputs['input_ids'].shape[1]
    n_tokens_total = outputs.shape[1]
    n_tokens_generados = n_tokens_total - n_tokens_prompt

    print(f" - Tokens del prompt: {n_tokens_prompt}")
    print(f" - Tokens del output total: {n_tokens_total}")
    print(f" - Tokens generados: {n_tokens_generados}")

    respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_texto = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    respuesta = respuesta_completa[len(prompt_texto):].strip()

    print(" RESPUESTA COMPLETA: " + json.dumps(respuesta_completa))
    print(" RESPUESTA CRUDA: " + json.dumps(respuesta))

    resultado = parsear_respuesta(respuesta)

    # Fallback por fallo de parseo
    if resultado['fallback']:
        ratio = (mz_fallback - stats['mz_min']) / max(stats['mz_max'] - stats['mz_min'], 1)
        resultado['predicted_ccs'] = round(
            stats['ccs_min'] + ratio * (stats['ccs_max'] - stats['ccs_min']), 2
        )
        resultado['pred_type'] = 'heuristic_fallback'
        resultado['reasoning'] = "Heuristic fallback: parser found no valid number"
        return resultado

    # Clasificar la predicción
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

def test_prompts():
    global MODEL, TOKENIZER, DATOS_TRAIN, STATS

    # Ejemplos de la tabla
    test_cases = [
        {"smiles": "O=C([C@@H](NS(=O)(=O)c1ccc(cc1)Cl)Cc1c[nH]c2c1cccc2)NC1CCCC1", "mz": 446.13, "adduct": "[M+H]+"},
        {"smiles": "COc1cccc(c1)[C@@H]1N(Cc2ccc(cc2)F)C(=O)c2c([C@@H]1C(=O)O)cccc2", "mz": 406.1449, "adduct": "[M+H]+"},
        {"smiles": "OC(=O)/C=C/c1ccc(cc1)OC(F)(F)F", "mz": 231.0275, "adduct": "[M-H]-"},
        {"smiles": "O=C1N[C@@H]2[C@H](N1)[C@@H](SC2)CCCCC(=O)N1CCC(CC1)C(=O)Nc1ccc2c(c1)OCO2", "mz": 473.1864, "adduct": "[M-H]-"},
        {"smiles": "Clc1ccc(cc1)c1occ(n1)CSc1nnnn1CCc1cccs1", "mz": 426.022, "adduct": "[M+Na]+"},
        {"smiles": "N#CC1(CCCC1)NC(=O)CSc1ccc(cn1)S(=O)(=O)N1CCCCC1", "mz": 431.1182, "adduct": "[M+Na]+"},
        {"smiles": "Oc1cccc(c1)I", "mz": 218.9312, "adduct": "[M-H]-"},
        {"smiles": "CCOc1cc2CC(Oc2cc1NC(=O)CN1C(=O)CCOc2c1cccc2)C", "mz": 397.1758, "adduct": "[M+H]+"},
        {"smiles": "Cc1ccc(cc1)n1nnnc1SCC(=O)N1CCCC1", "mz": 326.1046, "adduct": "[M+Na]+"},
        {"smiles": "COc1ccc2c(c1)CCCC12NC(=O)N(C1=O)CN1CCN(CC1)c1ccccc1", "mz": 419.2089, "adduct": "[M-H]-"},
    ]

    prompts_func = {
        "simple": lambda s, m, a: construir_prompt_simple(s, a),
        "con conocimiento experto": lambda s, m, a: construir_prompt_ce(s, a),
        "con ejemplos": lambda s, m, a: construir_prompt_ejemplos(s, a, DATOS_TRAIN, n=5),
        "completo": lambda s, m, a: construir_prompt_completo(s, a, DATOS_TRAIN, n=5),
    }

    print("=" * 70)
    print("TEST DE PROMPTS")

    for i, caso in enumerate(test_cases, 1):
        smiles, mz, adduct = caso["smiles"], caso["mz"], caso["adduct"]
        print(f"{'='*70}")
        print(f"COMPUESTO {i} | m/z = {mz} | Aducto = {adduct}")
        print(f"SMILES: {smiles[:60]}...")
        print(f"{'='*70}")

        # Limpiar caché entre predicciones
        torch.cuda.empty_cache()  # por si acaso aunque estés en CPU
        if hasattr(MODEL, 'reset_cache'):
            MODEL.reset_cache()

        for nombre, fn_prompt in prompts_func.items():
            prompt = fn_prompt(smiles, mz, adduct)
            print(f"Prompt {nombre}: ")
            resultado = predecir_ccs(MODEL, TOKENIZER, prompt, mz, STATS, seleccionar_ejemplos(DATOS_TRAIN, smiles, adduct, n=5))
            fallback_str = " [FALLBACK]" if resultado["fallback"] else ""
            print(f" - CCS = {resultado['predicted_ccs']:.2f} Ų{fallback_str}")
            print(f" - Reasoning: {resultado['reasoning'][:80]}")

    print(f"\n{'='*70}")
    print("TEST COMPLETADO")
    print("=" * 70)


# Para archivar resultados
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"../data/results/log_{timestamp}.txt"
    sys.stdout = Logger(log_path)
    print(f"Log guardado en: {log_path}")

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    if inicializar_app():
        test_prompts()
    else:
        print("ERROR en la inicialización. Verifica la configuración.")

    sys.stdout.close()
    sys.stdout = sys.stdout.terminal
