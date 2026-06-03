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


def leer_csv(filepath):
    datos = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                datos.append({
                    'smiles': row['smiles'].strip(),
                    'adduct': row['Adduct'].strip(),
                    'ccs': float(row['CCS_AVG'])
                })
            except (ValueError, KeyError):
                continue
    return datos


def analizar_datos(datos):
    ccs_values = [d['ccs'] for d in datos]
    stats = {
        'ccs_min': min(ccs_values),
        'ccs_max': max(ccs_values),
        'ccs_avg': sum(ccs_values) / len(ccs_values)
    }
    return stats


def normalizar_aducto(adduct):
    # Elimina la carga final del aducto para unificar formatos.
    return adduct.rstrip('+-').strip()

# Selecciona N moléculas del dataset estructuralmente similares al input.
def seleccionar_ejemplos(datos, smiles_input, adduct_input, n):

    # Paso 1: Calcula los descriptores del compuesto de entrada.
    caract_input = extraer_caracteristicas(smiles_input)
    adduct_norm = normalizar_aducto(adduct_input)

    # Paso 2: Filtra el training set por aducto.
    datos_filtrados = [d for d in datos if normalizar_aducto(d['adduct']) == adduct_norm]
    # Excepción: Si no tienes suficientes ejemplos del mismo aducto, se usa el dataset completo.
    if len(datos_filtrados) < n:
        print(f"Aviso: solo {len(datos_filtrados)} ejemplos para aducto {adduct_norm}, usando dataset completo")
        datos_filtrados = datos

    similitudes = []
    for d in datos_filtrados:
        caract = extraer_caracteristicas(d['smiles'])

        # Paso 3: Calcula una puntuación de similitud para cada descritor estructural.
        #  - Para cada molécula del conjunto filtrado
        #  - Compara los descriptores de la molecula con los del compuesto de entrada
        #  - Calcula una puntuación de 0 a 1
        sim_longitud   = 1 / (1 + abs(caract['length']    - caract_input['length']) / 10)
        sim_anillos    = 1 / (1 + abs(caract['num_rings'] - caract_input['num_rings']))
        sim_aromatico  = 1 / (1 + abs(caract['aromatic_atoms'] - caract_input['aromatic_atoms']) / 3)

        # Paso 4: Combina las tres puntuaciones con pesos (estructura > longitud > aromaticidad).
        similitud = 0.50 * sim_anillos + 0.30 * sim_longitud + 0.20 * sim_aromatico
        similitudes.append((similitud, d))

    # Paso 5: Ordena por puntuación y devuelve los 'n' mejores.
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
    '[M-H]':     {'charge': -1, 'mass_add': -1.007, 'effect': 'deprotonated, ~2-5 Å² smaller than [M+H]+'},
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

You MUST end your answer with the line "Final CCS: <number>", do not leave the number blank.
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

You MUST end your answer with the line "Final CCS: <number>", do not leave the number blank.
<think>
Let me apply the 4 steps to estimate the CCS.

Step 1 (structure): """
    return prompt

# Iteración 3: Monta un prompt usando ejemplos de una base de datos y la información del compuesto a analizar
def construir_prompt_ejemplos(smiles, adduct, DATOS_TRAIN, n):
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {
        'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'
    })

    # Seleccionar N moléculas estructuralmente similares del dataset
    ejemplos = seleccionar_ejemplos(DATOS_TRAIN, smiles, adduct, n)

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

You MUST end your answer with the line "Final CCS: <number>", do not leave the number blank.
<think>
Let me apply the 5 steps to estimate the CCS.

Step 1 (size): """
    return prompt

# Iteración 4: Monta un prompt usando conocimiento experto, ejemplos de una base de datos y la información del compuesto a analizar
def construir_prompt_completo(smiles, adduct, DATOS_TRAIN, n):
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {
        'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'
    })

    # Seleccionar N moléculas estructuralmente similares del dataset
    ejemplos = seleccionar_ejemplos(DATOS_TRAIN, smiles, adduct, n)

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

You MUST end your answer with the line "Final CCS: <number>", do not leave the number blank.
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
    texto = texto.strip() # Normalizamos espacios

    # Comprobación del bloque <think>
    if '<think>' in texto and '</think>' not in texto:
        return {'predicted_ccs': None, 'fallback': True, 'source': 'think_unclosed'}
    if '</think>' not in texto:
        return {'predicted_ccs': None, 'fallback': True, 'source': 'no_think_tag'}

    # A partir de aquí trabajamos SOLO con la zona post-think
    texto_post = texto.split('</think>')[-1].strip()

    def aceptar(valor, source):
        if 50 <= valor <= 500: # rango razonable para CCS
            return {'predicted_ccs': round(valor, 2), 'fallback': False, 'source': source}
        return None

    NUM = r'([0-9]+(?:\.[0-9]+)?)'

    # ESTRATEGIA 1: "Final CCS: <número>" en variantes (con/sin \boxed{})
    patrones_final = [
        rf'Final\s+CCS\s*[:=]?\s*\\?boxed\{{?\s*{NUM}\s*\}}?', # "Final CCS: \boxed{200}" o "Final CCS: 200"
        rf'Final\s+CCS\s*[:=]\s*{NUM}', # "Final CCS: 200" o "Final CCS = 200"
        rf'Final\s+CCS\s+{NUM}' # "Final CCS 200" (sin separador)
    ]
    for patron in patrones_final:
        match = re.search(patron, texto_post, re.IGNORECASE)
        if match:
            valor = float(match.group(1))
            result = aceptar(valor, 'final_ccs_tag')
            if result:
                return result

    # ESTRATEGIA 2: \boxed{<número>} sin "Final CCS:" delante
    match = re.search(rf'\\?boxed\{{?\s*{NUM}\s*\}}?', texto_post)
    if match:
        valor = float(match.group(1))
        result = aceptar(valor, 'boxed_tag')
        if result:
            return result

    # ESTRATEGIA 3: último número plausible en el rango CCS
    matches = re.findall(r'\b([0-9]{2,3}(?:\.[0-9]+)?)\b', texto_post)
    plausibles = [float(m) for m in matches if 50 <= float(m) <= 500]
    if plausibles:
        return {
            'predicted_ccs': round(plausibles[-1], 2),
            'fallback': False,
            'source': 'last_plausible',
        }

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


def predecir_ccs(model, tokenizer, prompt, stats, ejemplos):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=5000)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample = False,           # 1.5B -> False ; 7B/14B -> True # True para el 7B
            temperature = None,           # 1.5B -> None  ; 7B/14B -> 0.3 # Bajo para mantener precisión
            top_p = None,                 # 1.5B -> None  ; 7B/14B -> 0.9
            repetition_penalty = 1.15,   # 1.5B -> None  ; 7B/14B -> 1.15 # Penaliza repeticiones
            max_new_tokens = 10000, # más margen para el bloque <reasoning>
            pad_token_id = tokenizer.eos_token_id
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
        resultado['predicted_ccs'] = 0.0
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

    # Cargar modelo
    print("Cargando modelo DeepSeek...")
    MODEL, TOKENIZER = cargar_modelo()
    print(" - Modelo cargado y listo")
    print(" - Aplicación lista para recibir peticiones")
    print("=" * 70)
    return True



from openpyxl import Workbook, load_workbook

def inicializar_excel(filepath, variantes):
    """Crea el Excel con cabeceras si no existe."""
    if os.path.exists(filepath):
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "Predicciones CCS"
    cabeceras = ["", "SMILES", "Aducto"]
    cabeceras.extend(variantes)
    ws.append(cabeceras)
    wb.save(filepath)
    print(f" - Excel creado: {filepath}")

def guardar_fila_excel(filepath, fila):
    """Añade una fila al Excel y guarda inmediatamente."""
    wb = load_workbook(filepath)
    ws = wb.active
    ws.append(fila)
    wb.save(filepath)

def actualizar_celda_excel(filepath, fila_idx, col_idx, valor):
    """Actualiza una celda concreta y guarda inmediatamente."""
    wb = load_workbook(filepath)
    ws = wb.active
    ws.cell(row=fila_idx, column=col_idx, value=valor)
    wb.save(filepath)


def test_prompts():

    global MODEL, TOKENIZER, DATOS_TRAIN, STATS

    # 390 test cases extracted from test.csv
    # Total: 390 compounds ([M+H]: 66+64=130, [M-H]: 67+63=130, [M+Na]: 58+72=130)
    # Nuevos compuestos añadidos para equilibrar el test bench (199 compuestos)
    test_cases = [
        # [M+H]+ cases (+64)
        {"smiles": "COc1ccccc1N(S(=O)(=O)c1ccc(cc1)Cl)Cc1onc(n1)c1ccsc1", "adduct": "[M+H]+", "real_ccs": 199.79},
        {"smiles": "Cc1onc(c1)C(=O)N(C1CC1)Cc1nnc(o1)c1ccccc1Cl", "adduct": "[M+H]+", "real_ccs": 183.08},
        {"smiles": "NCCC(c1ccccc1)c1ccccc1", "adduct": "[M+H]+", "real_ccs": 154.72},
        {"smiles": "O=C(N1N=C(CC1c1ccco1)c1cccs1)Cn1nnc(n1)c1ccccc1C", "adduct": "[M+H]+", "real_ccs": 205.87},
        {"smiles": "c1ccc(cc1)Cn1nc(c(c1)CNC1CC1)c1cccnc1", "adduct": "[M+H]+", "real_ccs": 175.12},
        {"smiles": "O=C(NCC1(CC1)c1ccccc1)CCCn1cnc2c(c1=O)cccc2", "adduct": "[M+H]+", "real_ccs": 182.64},
        {"smiles": "COc1ccc(cc1)c1cc(ccc1OC)NC(=O)Cc1c(C)noc1C", "adduct": "[M+H]+", "real_ccs": 197.87},
        {"smiles": "COc1cccc(c1)C(=O)N1CCN(CC1)Cc1csc(n1)C1CCCCC1", "adduct": "[M+H]+", "real_ccs": 205.96},
        {"smiles": "COC(=O)C1=C(C)NC(=C(C1C(=O)OCC(=O)Nc1onc(c1)C(C)(C)C)C(=O)OC)C", "adduct": "[M+H]+", "real_ccs": 211.85},
        {"smiles": "CCOc1ccc(c(c1)O)[C@H]1N(Cc2ccccc2)C(=O)c2c([C@H]1C(=O)O)cccc2", "adduct": "[M+H]+", "real_ccs": 196.91},
        {"smiles": "CC1CCCCN1C(=O)c1ccc(cc1)S(=O)(=O)N1CCCCC1C", "adduct": "[M+H]+", "real_ccs": 193.91},
        {"smiles": "NC(=O)CSc1nnc(n1c1ccccc1Cl)N1CCCC1", "adduct": "[M+H]+", "real_ccs": 172.91},
        {"smiles": "O=C(N1CCCC1c1ccc2c(c1)OCCCO2)CCc1c(C)nc2n(c1C)nc(n2)C(F)(F)F", "adduct": "[M+H]+", "real_ccs": 197.47},
        {"smiles": "CCOC(=O)c1c(C)[nH]c(c1C)C(=O)OCc1csc(n1)COc1ccc(cc1)F", "adduct": "[M+H]+", "real_ccs": 205.39},
        {"smiles": "CN(C(=O)c1cc(ccc1N1CCCC1)S(=O)(=O)N(C)C)Cc1c(C)nn(c1C)c1ccccc1", "adduct": "[M+H]+", "real_ccs": 211.94},
        {"smiles": "O=C(NC(c1cccs1)c1ccccc1)CSc1nccn1Cc1ccccc1", "adduct": "[M+H]+", "real_ccs": 206.72},
        {"smiles": "O=C(c1ccc(cc1)n1cnc2c1cccc2)NCC(c1ccco1)N1CCCC1", "adduct": "[M+H]+", "real_ccs": 209.53},
        {"smiles": "COCCOC(=O)C1=C(C)N(C(=O)NC1c1ccc(cc1)C#N)C(COC)C", "adduct": "[M+H]+", "real_ccs": 197.34},
        {"smiles": "N#Cc1ccccc1c1ccc(cc1)COC(=O)[C@H](Cc1c[nH]c2c1cccc2)NC(=O)C", "adduct": "[M+H]+", "real_ccs": 205.9},
        {"smiles": "Brc1ccc(o1)C(=O)Nc1ccccc1OC(F)(F)F", "adduct": "[M+H]+", "real_ccs": 162.69},
        {"smiles": "O=C(C1CCCN1S(=O)(=O)c1ccc(cc1)OC(F)(F)F)N(C)C", "adduct": "[M+H]+", "real_ccs": 176.02},
        {"smiles": "OC(=O)c1ccc(cc1)OCC(F)(F)F", "adduct": "[M+H]+", "real_ccs": 147.15},
        {"smiles": "N#Cc1c(C)c(n(c1NC(=O)CN1CCCC1c1cccs1)CC1CCCO1)C", "adduct": "[M+H]+", "real_ccs": 199.99},
        {"smiles": "CC(NC(=O)CCn1cnc2c(c1=O)c1CCCCc1s2)C", "adduct": "[M+H]+", "real_ccs": 181.23},
        {"smiles": "COc1cc(cc(c1OC)OC)c1nnc(o1)SCC(=O)c1cc(n(c1C)CC1CCCO1)C", "adduct": "[M+H]+", "real_ccs": 232.58},
        {"smiles": "O=C(N1CCC(CC1)Cc1ccccc1)Cn1c(C)nc2c(c1=O)cccc2", "adduct": "[M+H]+", "real_ccs": 199.3},
        {"smiles": "O=C(Cc1c(C)noc1C)NCC(c1c[nH]c2c1cccc2)c1ccccc1Cl", "adduct": "[M+H]+", "real_ccs": 193.28},
        {"smiles": "COC(=O)c1c(C)[nH]c(c1C)C(=O)NCC12CC3CC(C2)CC(C1)C3", "adduct": "[M+H]+", "real_ccs": 191.47},
        {"smiles": "CCOCCCNCC(=O)N1N=C(CC1c1ccc(cc1)OC)c1ccc2c(c1)cccc2", "adduct": "[M+H]+", "real_ccs": 225.39},
        {"smiles": "O=C(N1CCCC1c1cccs1)Cn1nnn(c1=O)c1ccccc1", "adduct": "[M+H]+", "real_ccs": 182.7},
        {"smiles": "CNC(=O)COC(=O)c1c(C)nn(c1C)c1ccccc1", "adduct": "[M+H]+", "real_ccs": 170.97},
        {"smiles": "CCS(=O)(=O)Nc1cccc(c1)Oc1ccccc1", "adduct": "[M+H]+", "real_ccs": 161.28},
        {"smiles": "Fc1ccc(c(c1)C(=O)Nc1nc2c(s1)cccc2)Br", "adduct": "[M+H]+", "real_ccs": 165.8},
        {"smiles": "Clc1ccc(cc1)c1cnc(o1)CN1CCN(CC1)c1nccs1", "adduct": "[M+H]+", "real_ccs": 192.1},
        {"smiles": "O=C(c1cc(nc2c1cccc2)c1ccncc1)NCc1ccc(cc1)S(=O)(=O)N1CCOCC1", "adduct": "[M+H]+", "real_ccs": 208.24},
        {"smiles": "CCC1(NC(=O)N(C1=O)CC(=O)N1CCCCC1C)c1ccc(cc1)F", "adduct": "[M+H]+", "real_ccs": 187.49},
        {"smiles": "CC1CC(C(=O)O1)Sc1nnc(n1c1ccccc1)c1ccc2c(c1)OCO2", "adduct": "[M+H]+", "real_ccs": 189.01},
        {"smiles": "O=C(NCc1ccc(nc1)n1cncc1)NOCc1ccccc1", "adduct": "[M+H]+", "real_ccs": 186.53},
        {"smiles": "O=C(C(c1ccccc1)Sc1nnc(o1)c1ccccc1F)N1CCCCC1", "adduct": "[M+H]+", "real_ccs": 193.05},
        {"smiles": "COc1ccccc1N(S(=O)(=O)c1ccc(cc1)C(=O)Nc1ncc(s1)C)C", "adduct": "[M+H]+", "real_ccs": 204.54},
        {"smiles": "CC(C(C(=O)NC1CC1)NS(=O)(=O)c1ccc2c(c1)OCCO2)C", "adduct": "[M+H]+", "real_ccs": 182.57},
        {"smiles": "O=C(c1n[nH]c(c1)c1ccccc1)N1CCN(CC1)S(=O)(=O)c1ccc2c(c1)OCCO2", "adduct": "[M+H]+", "real_ccs": 214.73},
        {"smiles": "OCC1CCCN(C1)CC(=O)NC(c1ccc2c(c1)OCCCO2)C(C)C", "adduct": "[M+H]+", "real_ccs": 200.28},
        {"smiles": "CCCC1(NC(=O)N(C1=O)CC(=O)NC12CC3CC(C2)CC(C1)C3)c1ccccc1", "adduct": "[M+H]+", "real_ccs": 209.4},
        {"smiles": "O=C(Nc1ccccc1C(=O)Nc1ccc2c(c1)OCO2)CSc1nnc(n1C1CC1)C1CC1", "adduct": "[M+H]+", "real_ccs": 218.98},
        {"smiles": "O=C(c1ccc2c(c1)c(on2)c1ccccc1)NC(c1ccccc1)CN1CCOCC1", "adduct": "[M+H]+", "real_ccs": 213.15},
        {"smiles": "O=C(c1ccc(=O)n(n1)C)NCC1(CCCCC1)N1CCCCC1", "adduct": "[M+H]+", "real_ccs": 182.23},
        {"smiles": "c1ccc(cc1)CSc1nnc([nH]1)c1cccs1", "adduct": "[M+H]+", "real_ccs": 158.78},
        {"smiles": "O=C(N1CCN(CC1)C(=O)c1ccc2c(c1)cccc2)CN1C(=O)NC2(C1=O)CCCC2", "adduct": "[M+H]+", "real_ccs": 204.26},
        {"smiles": "CC(CC(N1CCOCC1)CNC(=O)C1CCCO1)C", "adduct": "[M+H]+", "real_ccs": 169.52},
        {"smiles": "CCN(S(=O)(=O)c1cccc(c1)c1nnc(n1C)SCC(=O)c1ccc(cc1)F)CC", "adduct": "[M+H]+", "real_ccs": 210.89},
        {"smiles": "O=C(c1ccc2c(c1)CCCC2)CSc1nnc(o1)c1ccccc1Br", "adduct": "[M+H]+", "real_ccs": 198.17},
        {"smiles": "O=C(c1ccc(cc1)S(=O)(=O)N1CCCC1)Nc1ccnn1Cc1cccs1", "adduct": "[M+H]+", "real_ccs": 206.69},
        {"smiles": "O=C(c1ccc(nc1)C(F)(F)F)OCc1nnnn1c1ccccc1", "adduct": "[M+H]+", "real_ccs": 181.02},
        {"smiles": "CC([C@H](N1C(=O)c2c(C1=O)cccc2)C(=O)NCC(c1ccccc1)N(C)C)C", "adduct": "[M+H]+", "real_ccs": 197.89},
        {"smiles": "CN(Cc1cc(=O)n2c(n1)scc2)CC1COc2c(O1)cccc2", "adduct": "[M+H]+", "real_ccs": 178.05},
        {"smiles": "O=C(CC1Sc2ccccc2NC1=O)NCc1cccnc1", "adduct": "[M+H]+", "real_ccs": 174.77},
        {"smiles": "COc1cc2CN(CCc2cc1OC)S(=O)(=O)c1ccccc1Cl", "adduct": "[M+H]+", "real_ccs": 181.05},
        {"smiles": "Clc1ccc(cc1)OCc1nnc(o1)c1ccccc1", "adduct": "[M+H]+", "real_ccs": 169.36},
        {"smiles": "O=C(C1CCCN(C1)C(=O)c1ccco1)Nc1ccccc1N1CCCC1=O", "adduct": "[M+H]+", "real_ccs": 191.84},
        {"smiles": "O=C1N(Cc2nnc(o2)c2ccccc2Br)C(=O)C2C1CC=CC2", "adduct": "[M+H]+", "real_ccs": 179.1},
        {"smiles": "O=C(N1CCN(CC1)CC(=O)N1CCCC1)CCc1cccs1", "adduct": "[M+H]+", "real_ccs": 182.43},
        {"smiles": "O=C(C1CCCN1S(=O)(=O)c1c(C)noc1C)Nc1cccc(c1)S(=O)(=O)N(C)C", "adduct": "[M+H]+", "real_ccs": 206.94},
        {"smiles": "CC(C(=O)N1CCOCC1)Oc1ccc(cc1)C(=O)c1ccccc1", "adduct": "[M+H]+", "real_ccs": 188.98},

        # [M-H]- cases (+63)
        {"smiles": "O=C(CN1C(=O)CSc2c1cccc2)Nc1ccc(cc1)C(=O)N", "adduct": "[M-H]-", "real_ccs": 183.93},
        {"smiles": "Cc1cccc(c1)N(C(=O)Cn1nnc(n1)c1cccs1)Cc1ccccc1", "adduct": "[M-H]-", "real_ccs": 197.79},
        {"smiles": "Ic1cc(ccc1C)NC(=O)c1c(C)noc1C", "adduct": "[M-H]-", "real_ccs": 166.54},
        {"smiles": "CCOc1ccc(cc1)S(=O)(=O)N1CCNC(=O)C1", "adduct": "[M-H]-", "real_ccs": 169.52},
        {"smiles": "CCCCN(c1c(=O)[nH]c(=O)n(c1N)Cc1ccccc1)C(=O)CSc1nnc(o1)Cc1csc(n1)C", "adduct": "[M-H]-", "real_ccs": 222.4},
        {"smiles": "O=C(Cc1ccc(cc1)NC(=O)C)NCC(=O)Nc1ccc(c(c1F)F)F", "adduct": "[M-H]-", "real_ccs": 187.53},
        {"smiles": "O=C(NCc1ccc(cc1)NC(=O)C1CC1)NCCc1ccccc1F", "adduct": "[M-H]-", "real_ccs": 182.25},
        {"smiles": "O=C(N1CCc2c1cccc2)CSc1nnc([nH]1)c1ccccc1Br", "adduct": "[M-H]-", "real_ccs": 186.54},
        {"smiles": "Fc1ccc(cc1)NC(=O)C(n1cnc2c1c(=O)n(C)c(=O)n2C)C", "adduct": "[M-H]-", "real_ccs": 180.1},
        {"smiles": "COc1ccc(cc1S(=O)(=O)N(C)C)NS(=O)(=O)c1ccc2c(c1)CCCC2", "adduct": "[M-H]-", "real_ccs": 198.12},
        {"smiles": "N#CC(=c1[nH]nc2n1CCCCC2)C(=O)CSc1ncccn1", "adduct": "[M-H]-", "real_ccs": 182.1},
        {"smiles": "CCN(CCNC(=O)c1ccc(cc1)NC(=O)c1ccc(cc1)Br)CC", "adduct": "[M-H]-", "real_ccs": 206.12},
        {"smiles": "CCCN(C(=O)c1ccc(cc1)NC(=O)c1cccs1)Cc1nnc(o1)c1ccco1", "adduct": "[M-H]-", "real_ccs": 204.7},
        {"smiles": "O=C(NCC1COc2c(O1)cccc2)COc1ccc(cc1)c1ccccc1", "adduct": "[M-H]-", "real_ccs": 207.85},
        {"smiles": "CSc1ccc(cc1)NC(=O)C12CC3CC(C1)CC(C2)(C3)O", "adduct": "[M-H]-", "real_ccs": 183.4},
        {"smiles": "CSCCC(C(=O)NCc1cccnc1)NC(=O)c1ccccc1Br", "adduct": "[M-H]-", "real_ccs": 198.29},
        {"smiles": "O=C(Nc1ccccc1)CCN1C(=O)CCC1=O", "adduct": "[M-H]-", "real_ccs": 154.07},
        {"smiles": "COc1ccc(cc1)NC(=O)CN(c1nc2cc(nn2c2c1cccc2)C)C", "adduct": "[M-H]-", "real_ccs": 200.7},
        {"smiles": "CCOc1ccc(cc1)S(=O)(=O)N(c1ccc(cc1)OCC(=O)Nc1scc(n1)c1cc(OC)ccc1OC)C", "adduct": "[M-H]-", "real_ccs": 227.46},
        {"smiles": "O=C(C1CCCN1S(=O)(=O)c1c(C)noc1C)Nc1ccc(c(c1)Cl)C", "adduct": "[M-H]-", "real_ccs": 186.01},
        {"smiles": "O=C(C1CN(C(=O)C1)c1ccc2c(c1)OCCO2)Nc1ccccc1N1CCOCC1", "adduct": "[M-H]-", "real_ccs": 209.65},
        {"smiles": "Cc1ccc(c(c1)CN1C(=O)NC(C1=O)(C)c1ccccc1Cl)C", "adduct": "[M-H]-", "real_ccs": 179.84},
        {"smiles": "O=C(Nc1sc(c(c1C(=O)N1CCOCC1)C)c1ccccc1)CN1CCc2c(C1)cccc2", "adduct": "[M-H]-", "real_ccs": 221.76},
        {"smiles": "CC1OCCN(C1)Cn1nc(n(c1=S)c1ccccc1)c1ccncc1", "adduct": "[M-H]-", "real_ccs": 186.06},
        {"smiles": "C=CCn1c(SCC(=O)c2ccc(s2)Cl)nnc1COc1ccccc1F", "adduct": "[M-H]-", "real_ccs": 193.27},
        {"smiles": "Cc1ccc(s1)CN(C(=O)Cn1cnc2c1c(=O)n(C)c(=O)n2C)CCc1ccccc1", "adduct": "[M-H]-", "real_ccs": 222.01},
        {"smiles": "CC(c1nnc(o1)c1ccc(cc1)[N+](=O)[O-])Sc1nnc(o1)c1ccc2c(c1)OCO2", "adduct": "[M-H]-", "real_ccs": 201.22},
        {"smiles": "O=S(=O)(c1cccc(c1)c1nnc(o1)SCCc1ccccc1)N1CCOCC1", "adduct": "[M-H]-", "real_ccs": 202.74},
        {"smiles": "CCN(C(=O)C1CCN(CC1)c1ccc2n(n1)nnn2)CC", "adduct": "[M-H]-", "real_ccs": 139.74},
        {"smiles": "O=C(C1CCN(CC1)S(=O)(=O)c1c(C)noc1C)NC1CCCCCC1", "adduct": "[M-H]-", "real_ccs": 203.29},
        {"smiles": "CCN1C(=O)C(=O)N(C1=O)CC(=O)c1ccc(cc1)C(C)C", "adduct": "[M-H]-", "real_ccs": 176.56},
        {"smiles": "COc1ccc(cc1)c1noc(c1)Cn1nnc(n1)c1ccc(cc1)Cl", "adduct": "[M-H]-", "real_ccs": 192.04},
        {"smiles": "Brc1ccc(cc1)c1ccc(cc1)OS(=O)(=O)c1ccc2c(c1)oc(=O)n2C", "adduct": "[M-H]-", "real_ccs": 203.85},
        {"smiles": "COc1cc2CN(CCc2cc1OC)CC(COCC1CCCO1)O", "adduct": "[M-H]-", "real_ccs": 181.43},
        {"smiles": "COc1ccccc1CSc1nc2c([nH]1)cc(cc2)Cl", "adduct": "[M-H]-", "real_ccs": 169.46},
        {"smiles": "O=C(C1CCN(CC1)c1ccc2n(n1)c(nn2)C(F)(F)F)NCCc1ccc2c(c1)OCCO2", "adduct": "[M-H]-", "real_ccs": 212.17},
        {"smiles": "CCCCn1ccnc1C(C(F)(F)F)O", "adduct": "[M-H]-", "real_ccs": 176.79},
        {"smiles": "C=CCn1c(SCC(=O)NC(=O)Nc2ccc3c(c2)OCCO3)nnc1C", "adduct": "[M-H]-", "real_ccs": 195.81},
        {"smiles": "CCCCc1ccc(cc1)NC(=O)CNC(=O)c1ccc(c(c1)OC)OC", "adduct": "[M-H]-", "real_ccs": 213.39},
        {"smiles": "COC(=O)CNC(=O)C1CCCN1S(=O)(=O)c1cc(C)c(cc1C)Cl", "adduct": "[M-H]-", "real_ccs": 191.09},
        {"smiles": "O=C(N1CCCC1c1nc2c(s1)cccc2)c1cccc(c1)S(=O)(=O)NCC1CCCO1", "adduct": "[M-H]-", "real_ccs": 218.17},
        {"smiles": "O=C(c1ccc(c(c1)S(=O)(=O)Nc1ccccc1N1CCOCC1)Cl)N1CCOCC1", "adduct": "[M-H]-", "real_ccs": 199.49},
        {"smiles": "Clc1c(cnn(c1=O)c1ccccc1)N1CCN(CC1)Cc1ccc2c(c1)OCO2", "adduct": "[M-H]-", "real_ccs": 203.5},
        {"smiles": "O=C(C(Sc1nc2ccsc2c(=O)n1C)C)NC1CCCCCC1", "adduct": "[M-H]-", "real_ccs": 198.67},
        {"smiles": "O=C(Nc1nnc(s1)c1cccnc1)COc1ccc(cc1)Cl", "adduct": "[M-H]-", "real_ccs": 182.59},
        {"smiles": "CCOc1ccc(cc1)C(=O)N[C@H](C(=O)Nc1cccc(c1)OC)C(C)C", "adduct": "[M-H]-", "real_ccs": 201.76},
        {"smiles": "CC1CC(C)(C)CC2(C1)NC(=O)N(C2=O)CC(=O)Nc1ccc(c(c1F)F)F", "adduct": "[M-H]-", "real_ccs": 191.68},
        {"smiles": "O=C(CN1C(=O)NC2(C1=O)CCc1c2cccc1)NC1CCCCC1", "adduct": "[M-H]-", "real_ccs": 185.1},
        {"smiles": "CCc1cccc(c1NC(=O)C1CN(C(=O)C1)Cc1ccccc1)CC", "adduct": "[M-H]-", "real_ccs": 190.42},
        {"smiles": "O=C([C@@H]1CCCN1S(=O)(=O)c1ccccc1F)Nc1ccc(c(c1)F)C", "adduct": "[M-H]-", "real_ccs": 178.49},
        {"smiles": "CC(NC(=O)COC(=O)c1c(C)c(nc2c1cccc2)c1cccc(c1)Cl)C", "adduct": "[M-H]-", "real_ccs": 209.77},
        {"smiles": "O=C(COc1ccc(cc1)c1ccccc1)NCC(c1ccc(cc1)F)N1CCOCC1", "adduct": "[M-H]-", "real_ccs": 217.81},
        {"smiles": "O=C(NC12CC3CC(C2)CC(C1)C3)CN1C(=O)N(C(C1=O)C)c1ccc(cc1)C", "adduct": "[M-H]-", "real_ccs": 211.38},
        {"smiles": "O=C(CN1C(=O)NC2(C1=O)CCCCC2)NC(=O)NC(C)(C)C", "adduct": "[M-H]-", "real_ccs": 185.66},
        {"smiles": "CCOc1cc(NC(=O)c2ccccc2)c(cc1NC(=O)Cn1ccccc1=O)OCC", "adduct": "[M-H]-", "real_ccs": 215.3},
        {"smiles": "CCCCCCn1nc(C(=O)OCc2cc(=O)oc3c2ccc(c3)OC)c2c(c1=O)cccc2", "adduct": "[M-H]-", "real_ccs": 223.73},
        {"smiles": "COc1ccccc1C1CC(=NN1S(=O)(=O)C)c1ccccc1", "adduct": "[M-H]-", "real_ccs": 180.01},
        {"smiles": "O=C(C1=C(C)NC(=S)NC1c1ccc(cc1)C)N1CCOCC1", "adduct": "[M-H]-", "real_ccs": 179.32},
        {"smiles": "COc1cccc(c1)c1nnn(n1)c1nc(CN2CCOCC2)nc2c1c(C)c(s2)C", "adduct": "[M-H]-", "real_ccs": 206.66},
        {"smiles": "CCN1C(=O)CC(=O)N(C1=S)CC", "adduct": "[M-H]-", "real_ccs": 137.02},
        {"smiles": "CC(=O)Nc1ccc(cc1)NC(c1nnc(o1)c1ccccc1)C", "adduct": "[M-H]-", "real_ccs": 177.68},
        {"smiles": "O=C(NC(C12CC3CC(C2)CC(C1)C3)C)CSc1nnc(o1)c1cccc(c1)S(=O)(=O)N1CCCCC1", "adduct": "[M-H]-", "real_ccs": 240.78},
        {"smiles": "Cc1ccc(c(c1)C)CN(C(=O)Cc1c[nH]c2c1cccc2)C", "adduct": "[M-H]-", "real_ccs": 178.31},

        # [M+Na]+ cases (+72)
        {"smiles": "COc1ccc(cc1)n1c2nnc(n2c2c(c1=O)cccc2)SCC(=O)N1CCCCC1C", "adduct": "[M+Na]+", "real_ccs": 220.44},
        {"smiles": "Brc1ccc(s1)c1nnc(o1)CN1C(=O)NC(C1=O)(C)c1ccc2c(c1)OCO2", "adduct": "[M+Na]+", "real_ccs": 214.22},
        {"smiles": "COc1cc2CN(CCc2cc1OC)C(=O)CC12CC3CC(C2)CC(C1)C3", "adduct": "[M+Na]+", "real_ccs": 208.0},
        {"smiles": "O=c1cc(CN2C(=O)NC(C2=O)(C)c2ccc3c(c2)OCO3)c2c(o1)cc1c(c2)CCC1", "adduct": "[M+Na]+", "real_ccs": 207.76},
        {"smiles": "O=C(N1CCSC1c1ccccc1F)CSc1nncn1c1ccccc1", "adduct": "[M+Na]+", "real_ccs": 188.19},
        {"smiles": "Cc1ccc(cc1)c1nnn(n1)CC(=O)c1cc(n(c1C)CC1CCCO1)C", "adduct": "[M+Na]+", "real_ccs": 208.02},
        {"smiles": "O=C(N(Cc1ccccc1)Cc1ccccc1)COc1ccc(cc1)c1nnco1", "adduct": "[M+Na]+", "real_ccs": 192.23},
        {"smiles": "O=C(NC(C(=O)C)Cc1ccccc1)CSc1nnc(n1C)c1ccccc1F", "adduct": "[M+Na]+", "real_ccs": 197.54},
        {"smiles": "COc1cc(cc(c1OC)OC)C(=O)NCC(=O)N1CCCCCCC1", "adduct": "[M+Na]+", "real_ccs": 202.5},
        {"smiles": "O=C(CN1C(=O)NC2(C1=O)CCCc1c2cccc1)NC1CCCCC1", "adduct": "[M+Na]+", "real_ccs": 195.33},
        {"smiles": "COc1ccc(cc1S(=O)(=O)NC1CC1)C(=O)Nc1ccc(cc1)c1ccccc1", "adduct": "[M+Na]+", "real_ccs": 199.54},
        {"smiles": "COc1cc(ccc1OC)C(=O)Nc1ccccc1C(=O)NCc1ccco1", "adduct": "[M+Na]+", "real_ccs": 195.18},
        {"smiles": "CN(Cc1ccccc1)CC(=O)Nc1ccc(cc1Cl)C", "adduct": "[M+Na]+", "real_ccs": 179.7},
        {"smiles": "COc1ccc(cc1S(=O)(=O)NC1CC1)C(=O)NCCC(C)C", "adduct": "[M+Na]+", "real_ccs": 196.31},
        {"smiles": "O=C1N(c2ccccc2)C(Nc2c1cccc2)c1c[nH]nc1c1cccnc1", "adduct": "[M+Na]+", "real_ccs": 188.24},
        {"smiles": "Fc1ccc(cc1)S(=O)(=O)N1CCCC1C(=O)N1CCN(CC1)S(=O)(=O)c1cccs1", "adduct": "[M+Na]+", "real_ccs": 205.05},
        {"smiles": "COc1cc(Br)cc2c1oc(=O)c(c2)C(=O)N1CCc2c(C1)cccc2", "adduct": "[M+Na]+", "real_ccs": 201.53},
        {"smiles": "COc1cc2CN(CCc2cc1OC)S(=O)(=O)c1ccccc1Cl", "adduct": "[M+Na]+", "real_ccs": 185.59},
        {"smiles": "OC(CN1C(=O)NC2(C1=O)CCCCC2)COc1c(C)cccc1C", "adduct": "[M+Na]+", "real_ccs": 192.34},
        {"smiles": "CC(=O)N1CCN(CC1)c1ccc(cc1)Oc1nc(nc2c1cccc2)c1cccnc1", "adduct": "[M+Na]+", "real_ccs": 207.71},
        {"smiles": "CC(=O)NC(C(=O)Nc1ccc(cc1)c1nnc2n1CCCCC2)C(C)C", "adduct": "[M+Na]+", "real_ccs": 198.82},
        {"smiles": "COc1ccc(cc1)N1CCN(CC1)CCC#N", "adduct": "[M+Na]+", "real_ccs": 163.44},
        {"smiles": "CCN(C(=O)CCNC(=O)N(Cc1ccc(cc1)c1ccccc1)C)CC", "adduct": "[M+Na]+", "real_ccs": 193.24},
        {"smiles": "O=C1NC2(C(=O)N1Cc1onc(n1)c1cccs1)CCOc1c2cccc1", "adduct": "[M+Na]+", "real_ccs": 188.89},
        {"smiles": "COc1cc(ccc1OC(F)F)C=C1C(=O)c2c(C1=O)cccc2", "adduct": "[M+Na]+", "real_ccs": 198.0},
        {"smiles": "Cc1sc2c(c1C)c(nc(n2)CN1CCOCC1)n1nnc(n1)c1ccccc1", "adduct": "[M+Na]+", "real_ccs": 218.31},
        {"smiles": "Cc1cccc(c1C)n1c(Sc2nnnn2c2ccccc2)nc2c(c1=O)cccc2", "adduct": "[M+Na]+", "real_ccs": 209.1},
        {"smiles": "CONC(=O)c1cc(nn1c1ccccc1)C1CC1", "adduct": "[M+Na]+", "real_ccs": 170.31},
        {"smiles": "O=C(Nc1cccc(c1C)C)CN(C(=O)CCc1ccc(cc1)C)C", "adduct": "[M+Na]+", "real_ccs": 185.41},
        {"smiles": "COc1cc(CC(=O)NC2CCCc3c2cccc3)cc(c1OC)OC", "adduct": "[M+Na]+", "real_ccs": 192.36},
        {"smiles": "COc1ccc(cc1OC)CNC(=O)CN(S(=O)(=O)c1ccc2c(c1)OCCO2)C", "adduct": "[M+Na]+", "real_ccs": 204.86},
        {"smiles": "Fc1ccc(cc1)C(N1CCOCC1)CNC(=O)c1cccnc1Sc1ccc(cc1)Cl", "adduct": "[M+Na]+", "real_ccs": 215.77},
        {"smiles": "O=C(N1CCN(CC1)C(=O)c1oc2c(c1COc1ccccc1)cccc2)C1CC1", "adduct": "[M+Na]+", "real_ccs": 193.85},
        {"smiles": "O=C(N1CCOCC1)CCCSc1nnc(n1c1ccccc1)COc1ccccc1F", "adduct": "[M+Na]+", "real_ccs": 215.79},
        {"smiles": "COc1ccc(cc1)S(=O)(=O)N1N=C(CC1c1ccc2c(c1)nccn2)c1cccs1", "adduct": "[M+Na]+", "real_ccs": 206.9},
        {"smiles": "CC(C(=O)NC1CCCC1)OC(=O)c1ccccc1", "adduct": "[M+Na]+", "real_ccs": 175.73},
        {"smiles": "CCOc1ccc(cc1)NC(=O)C(C(CC)C)NC(=O)c1ccccc1", "adduct": "[M+Na]+", "real_ccs": 204.96},
        {"smiles": "O=C1NC(C(=O)N1Cc1cccc(c1)S(=O)(=O)N1CCOCC1)(C)c1ccc2c(c1)OCO2", "adduct": "[M+Na]+", "real_ccs": 202.84},
        {"smiles": "Brc1ccccc1c1nnc(o1)Sc1ncnc2c1ccs2", "adduct": "[M+Na]+", "real_ccs": 189.73},
        {"smiles": "O=C(N1CCCCCCC1)CN1C(=O)NC(C1=O)(C)c1cc2c(o1)cccc2", "adduct": "[M+Na]+", "real_ccs": 199.05},
        {"smiles": "COc1cc(NC(=O)CC2CC3CC2CC3)cc(c1)OC", "adduct": "[M+Na]+", "real_ccs": 186.14},
        {"smiles": "CC=Cc1cc(cc(c1OC)OC)C(=O)NC1CC1", "adduct": "[M+Na]+", "real_ccs": 183.86},
        {"smiles": "O=C(N1CCN(CC1)c1ncccn1)C=Cc1ccc2c(c1)OCCCO2", "adduct": "[M+Na]+", "real_ccs": 202.52},
        {"smiles": "CN(CC(=O)Nc1ccccc1C(=O)C)CC(=O)Nc1ccc(cc1)C", "adduct": "[M+Na]+", "real_ccs": 201.61},
        {"smiles": "O=C(c1cnc2c(c1)cnn2C(C)C)Nc1sc2c(n1)CCC2", "adduct": "[M+Na]+", "real_ccs": 197.34},
        {"smiles": "O=C(N1CCN(CC1)c1ncc(cc1Cl)C(F)(F)F)COc1ccccc1C", "adduct": "[M+Na]+", "real_ccs": 209.61},
        {"smiles": "COC(=O)C(NC(=O)C(c1ccccc1)c1ccccc1)Cc1ccc(cc1)O", "adduct": "[M+Na]+", "real_ccs": 195.49},
        {"smiles": "CN(Cc1ccc(cc1)C(=O)N)CCOc1ccccc1", "adduct": "[M+Na]+", "real_ccs": 165.57},
        {"smiles": "O=C(N1CCN(CC1)S(=O)(=O)c1ccc2c(c1)CCC2)C1CCC1", "adduct": "[M+Na]+", "real_ccs": 191.0},
        {"smiles": "Cc1ccc(cc1)n1c(nn(c1=S)CN(CC(=O)N1CCCC1)C)c1ccc(cc1)Cl", "adduct": "[M+Na]+", "real_ccs": 202.08},
        {"smiles": "O=C(COc1ccccc1)NCC(=O)NC1CCCc2c1cccc2", "adduct": "[M+Na]+", "real_ccs": 178.28},
        {"smiles": "CCC(=O)N(c1nnc(s1)SCCOc1ccc(cc1)OC)C1CC1", "adduct": "[M+Na]+", "real_ccs": 185.31},
        {"smiles": "CCOc1cc(NC(=O)c2ccccc2)c(cc1NC(=O)Cn1ccccc1=O)OCC", "adduct": "[M+Na]+", "real_ccs": 210.97},
        {"smiles": "CC(C(=O)NCc1ccc2c(c1)OCO2)Sc1nnc(n1Cc1ccccc1)c1ccncc1", "adduct": "[M+Na]+", "real_ccs": 209.78},
        {"smiles": "O=C(NC(C)(C)C)CN1CCCN(CC1)c1ncnc2c1cc(s2)c1ccccc1", "adduct": "[M+Na]+", "real_ccs": 217.54},
        {"smiles": "O=C(Nc1c(C)n(n(c1=O)c1ccccc1)C)CSc1ccccc1C(=O)NC1=NCCS1", "adduct": "[M+Na]+", "real_ccs": 218.85},
        {"smiles": "N#Cc1c(NC(=O)C2CCCO2)sc2c1CCCC2", "adduct": "[M+Na]+", "real_ccs": 178.55},
        {"smiles": "COCc1c(sc2c1c(F)ccc2)C(=O)OCC(=O)c1cc(n(c1C)C(COC)C)C", "adduct": "[M+Na]+", "real_ccs": 211.94},
        {"smiles": "O=C(C1CN(C(=O)C1)Cc1ccco1)N1CCN(CC1)C(=O)c1cccs1", "adduct": "[M+Na]+", "real_ccs": 185.7},
        {"smiles": "COC(=O)C1(CCCCC1)NS(=O)(=O)N1CCCC(C1)C", "adduct": "[M+Na]+", "real_ccs": 182.34},
        {"smiles": "COC(=O)c1ccc(cc1)NC(=O)COC(=O)c1ccccc1c1ccc(cc1)C(F)(F)F", "adduct": "[M+Na]+", "real_ccs": 209.27},
        {"smiles": "O=C(Nc1sc(c(c1C(=O)N1CCOCC1)C)c1ccccc1)CSc1nncn1C", "adduct": "[M+Na]+", "real_ccs": 211.46},
        {"smiles": "O=C(CSc1ccccc1C(=O)Nc1cccc(c1)F)NCc1ccco1", "adduct": "[M+Na]+", "real_ccs": 190.67},
        {"smiles": "COc1ccc(cc1)OCc1nnc(n1c1ccccc1)SCc1nnc(o1)C", "adduct": "[M+Na]+", "real_ccs": 191.22},
        {"smiles": "CCn1c(Cn2nnc(n2)c2cscc2)nc2c1ccc(c2)S(=O)(=O)N1CCOCC1", "adduct": "[M+Na]+", "real_ccs": 224.57},
        {"smiles": "O=C(Nc1ccccc1C)CN1C(=O)NC(C1=O)(C)c1ccc(cc1)C(C)(C)C", "adduct": "[M+Na]+", "real_ccs": 209.39},
        {"smiles": "COc1cc(ccc1OCc1ccccc1)C(=O)N(Cc1ccccc1)C", "adduct": "[M+Na]+", "real_ccs": 192.64},
        {"smiles": "NC(=O)C1CCN(CC1)C(=O)c1cc(nn1c1ccccc1)c1ccccc1", "adduct": "[M+Na]+", "real_ccs": 198.5},
        {"smiles": "C=CCn1c(SCC(=O)N2CCC(CC2)C)nc2c(c1=O)c(cs2)c1ccco1", "adduct": "[M+Na]+", "real_ccs": 207.38},
        {"smiles": "CN(Cc1[nH]c(=O)c2c(n1)sc1c2CCC1)Cc1ccc(cc1)Cl", "adduct": "[M+Na]+", "real_ccs": 185.07},
        {"smiles": "O=C(c1cccc(c1)N1CCCC1=O)N1CCN(CC1)S(=O)(=O)c1ccc2c(c1)CCC2", "adduct": "[M+Na]+", "real_ccs": 209.45},
        {"smiles": "CCN1CCN(CC1)C1=NS(=O)(=O)c2c1cccc2", "adduct": "[M+Na]+", "real_ccs": 199.68},
    ]

    prompts_func = {
        "simple": lambda s, a: construir_prompt_simple(s, a),
        "con conocimiento experto": lambda s, a: construir_prompt_ce(s, a),
        "con ejemplos": lambda s, a: construir_prompt_ejemplos(s, a, DATOS_TRAIN, n=5),
        "completo": lambda s, a: construir_prompt_completo(s, a, DATOS_TRAIN, n=5),
    }

    variantes = list(prompts_func.keys())

    # Inicializar Excel con timestamp para no sobrescribir ejecuciones previas
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"../data/results/predicciones_prompts_{timestamp}.xlsx"
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    inicializar_excel(excel_path, variantes)

    print("=" * 70)
    print(f"TEST DE PROMPTS\nInicio de prueba: {datetime.now()}")

    for i, caso in enumerate(test_cases, 1):
        smiles, adduct = caso["smiles"], caso["adduct"]
        print(f"{'='*70}")
        print(f"COMPUESTO {i} | Aducto = {adduct}")
        print(f"SMILES: {smiles[:60]}...")
        print(f"{'='*70}")

        # Limpiar caché entre predicciones
        torch.cuda.empty_cache()  # por si acaso aunque estés en CPU
        if hasattr(MODEL, 'reset_cache'):
            MODEL.reset_cache()

        # Crear la fila base del compuesto (predicciones vacías, se rellenarán abajo)
        fila_base = [i, smiles, adduct] + [None] * len(variantes)
        guardar_fila_excel(excel_path, fila_base)
        # Cabecera = fila 1, así que el compuesto i va a fila i+1
        fila_idx = i + 1

        for j, (nombre, fn_prompt) in enumerate(prompts_func.items()):
            prompt = fn_prompt(smiles, adduct)
            print(f"Prompt {nombre}: ")
            print(f" - Hora de ejecución: {datetime.now().strftime('%H:%M:%S')}")

            ejemplos = seleccionar_ejemplos(DATOS_TRAIN, smiles, adduct, n=5)
            resultado = predecir_ccs(MODEL, TOKENIZER, prompt, STATS, ejemplos)

            fallback_str = " [FALLBACK]" if resultado["fallback"] else ""
            print(f" - CCS = {resultado['predicted_ccs']:.2f} Å²{fallback_str}")
            print(f" - Reasoning: {resultado['reasoning'][:80]}")

            # Guardar resultado inmediatamente en el Excel
            # Columnas: 1=id, 2=smiles, 3=aducto, 4..=variantes
            col_idx = 4 + j
            actualizar_celda_excel(excel_path, fila_idx, col_idx, resultado["predicted_ccs"])

    print(f"\n{'='*70}")
    print(f"TEST FINALIZADO!\nFin de prueba: {datetime.now()}")
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
