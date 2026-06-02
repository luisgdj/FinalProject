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


ADDUCT_INFO = {
    '[M+H]':     {'charge': 1,  'mass_add': 1.007,  'effect': 'protonated, baseline reference'},
    '[M+Na]':    {'charge': 1,  'mass_add': 22.989, 'effect': 'sodium adduct, ~3-5 Å² larger than [M+H]+'},
    '[M-H]':     {'charge': -1, 'mass_add': -1.007, 'effect': 'deprotonated, ~2-5 Å² smaller than [M+H]+'},
}

# Monta un prompt simple solo usando la información del compuesto a analizar
def construir_prompt(smiles, adduct):
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
def clasificar_prediccion(ccs_pred, stats):

    avg = round(stats['ccs_avg'])
    if abs(ccs_pred - avg) <= 0.5:
        return 'dataset_avg', False

    return 'interpolated', True

# SI VUELVO A EJECUTAR LAS PRUEBAS CAMBIAR:
#  - top_p = 0.95 (recomendado oficial)
#  - max_new_tokens = 20000 (tengo dos fallbacks con el modelo 1.5B)
#  - Investigar sobre el valor de la temperatura y entender porque he puesto 0.3 en vez de un valor dentro del rando 0.5-0.7 recomendado
def predecir_ccs(model, tokenizer, prompt, stats):

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
    tipo, es_valida = clasificar_prediccion(resultado['predicted_ccs'], stats)
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
    csv_path = r"../data/processed/other/train.csv"
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


def test():

    global MODEL, TOKENIZER, DATOS_TRAIN, STATS

    # 10 test cases extracted from train.csv
    test_cases = [
        # Original test cases
        {"smiles": "O=C([C@@H](NS(=O)(=O)c1ccc(cc1)Cl)Cc1c[nH]c2c1cccc2)NC1CCCC1", "adduct": "[M+H]+"},
        {"smiles": "COc1cccc(c1)[C@@H]1N(Cc2ccc(cc2)F)C(=O)c2c([C@@H]1C(=O)O)cccc2", "adduct": "[M+H]+"},
        {"smiles": "OC(=O)/C=C/c1ccc(cc1)OC(F)(F)F", "adduct": "[M-H]-"},
        {"smiles": "O=C1N[C@@H]2[C@H](N1)[C@@H](SC2)CCCCC(=O)N1CCC(CC1)C(=O)Nc1ccc2c(c1)OCO2", "adduct": "[M-H]-"},
        {"smiles": "Clc1ccc(cc1)c1occ(n1)CSc1nnnn1CCc1cccs1", "adduct": "[M+Na]+"},
        {"smiles": "N#CC1(CCCC1)NC(=O)CSc1ccc(cn1)S(=O)(=O)N1CCCCC1", "adduct": "[M+Na]+"},
        {"smiles": "Oc1cccc(c1)I", "adduct": "[M-H]-"},
        {"smiles": "CCOc1cc2CC(Oc2cc1NC(=O)CN1C(=O)CCOc2c1cccc2)C", "adduct": "[M+H]+"},
        {"smiles": "Cc1ccc(cc1)n1nnnc1SCC(=O)N1CCCC1", "adduct": "[M+Na]+"},
        {"smiles": "COc1ccc2c(c1)CCCC12NC(=O)N(C1=O)CN1CCN(CC1)c1ccccc1", "adduct": "[M-H]-"}
    ]

    print("=" * 70)
    print("PRUEBA DE EFICIENCIA\nInicio de prueba: {datetime.now()}")

    for i, caso in enumerate(test_cases, 1):
        smiles, adduct = caso["smiles"], caso["adduct"]
        print(f"{'='*70}")
        print(f"COMPUESTO {i} | Aducto = {adduct}")
        print(f"SMILES: {smiles}")
        print(f"{'='*70}")

        # Limpiar caché entre predicciones
        torch.cuda.empty_cache()  # por si acaso aunque estés en CPU
        if hasattr(MODEL, 'reset_cache'):
            MODEL.reset_cache()

        prompt = construir_prompt(smiles, adduct)
        resultado = predecir_ccs(MODEL, TOKENIZER, prompt, STATS)

        fallback_str = " [FALLBACK]" if resultado["fallback"] else ""
        print(f" CCS = {resultado['predicted_ccs']:.2f} Å²{fallback_str}")
        print(f" Reasoning: {resultado['reasoning']}")

    print(f"\n{'='*70}")
    print(f"TEST FINALIZADO!\nFin de prueba: {datetime.now()}")
    print("=" * 70)

if __name__ == '__main__':
    if inicializar_app():
        test()
    else:
        print("ERROR en la inicialización. Verifica la configuración.")
