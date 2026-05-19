from flask import Flask, render_template, request, jsonify
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
        'ccs_avg': sum(ccs_values) / len(ccs_values),
        'ccs_std': calcular_std(ccs_values),
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

# Monta un prompt usando conocimiento experto, ejemplos de una base de datos y la información del compuesto a analizar
def construir_prompt(smiles, adduct, DATOS_TRAIN, ejemplos):
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {
        'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'
    })

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

Below are {len(ejemplos)} reference molecules with experimentally measured CCS values, selected for their structural similarity to the target. Use them to identify patterns and infer the CCS of the target molecule.

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
        rf'Final\s+CCS\s+([0-9]+{NUM}' # "Final CCS 200" (sin separador)
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



@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global MODEL, TOKENIZER, DATOS_TRAIN, STATS

    try:
        data = request.json
        smiles = data.get('smiles', '').strip()
        adduct = data.get('adduct', '[M+H]+').strip()

        if not smiles:
            return jsonify({'error': 'Invalid input parameters'}), 400

        # Buscar si existe en el dataset
        ccs_real = buscar_en_dataset(smiles, adduct, DATOS_TRAIN)
        if ccs_real is not None:
            return jsonify({
                'success': True,
                'predicted_ccs': round(ccs_real, 2),
                'shape': 'Known',
                'reasoning': 'Exact match found in training dataset',
                'fallback': False,
                'from_dataset': True,
                'molecular_features': extraer_caracteristicas(smiles)
            })

        # Seleccionar 'n' moléculas estructuralmente similares del dataset
        ejemplos = seleccionar_ejemplos(DATOS_TRAIN, smiles, adduct, n=5)

        # Construir prompt
        prompt = construir_prompt(smiles, adduct, DATOS_TRAIN, ejemplos)

        # Predecir
        resultado = predecir_ccs(MODEL, TOKENIZER, prompt, STATS, ejemplos)

        # Añadir información adicional
        resultado['molecular_features'] = extraer_caracteristicas(smiles)
        resultado['similar_compounds'] = [
            {'smiles': ej['smiles'], 'ccs': ej['ccs'], 'adduct': ej['adduct']}
            for ej in ejemplos[:3]
        ]
        resultado['success'] = True
        resultado['from_dataset'] = False

        return jsonify(resultado)

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/status')
def status():
    return jsonify({
        'model_loaded': MODEL is not None,
        'dataset_loaded': DATOS_TRAIN is not None,
        'dataset_size': len(DATOS_TRAIN) if DATOS_TRAIN else 0,
        'stats': STATS if STATS else {}
    })


def inicializar_app():
    global MODEL, TOKENIZER, DATOS_TRAIN, STATS

    print("=" * 70)
    print("Inicializando aplicación de predicción CCS.")
    print("=" * 70)

    # Cargar datos
    csv_path = r"data/processed/train.csv"
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
    print(f" - CCS range: {STATS['ccs_min']:.2f} - {STATS['ccs_max']:.2f} Å²")

    # Cargar modelo
    print("Cargando modelo DeepSeek...")
    MODEL, TOKENIZER = cargar_modelo()
    print(" - Modelo cargado y listo")
    print(" - Aplicación lista para recibir peticiones")
    print("=" * 70)
    return True


if __name__ == '__main__':
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    if inicializar_app():
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        print("ERROR en la inicialización. Verifica la configuración.")
