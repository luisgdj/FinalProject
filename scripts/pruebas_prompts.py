from flask import Flask, render_template, request, jsonify
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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


def seleccionar_ejemplos(datos, smiles_input, mz_input, adduct_input, n = 10):
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
        sim_mz = 1/(1 + abs(d['mz'] - mz_input) / 100) # Mide la proximidad en masas, tiene un peso del 35%
        sim_longitud = 1/(1 + abs(caract['length'] - caract_input['length']) / 10) # Mide la longitud del SMILES, tiene un peso del 15%
        sim_estructura = 1/(1 + abs(caract['num_rings'] - caract_input['num_rings'])) # Mide el número de anillos, tiene un peso del 50%

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


def construir_prompt_completo(smiles, mz, adduct, stats, ejemplos, n = 5):
    feat = extraer_caracteristicas(smiles)

    # Información del aducto
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'})

    ejemplos_texto = ""
    for ej in ejemplos[:n]:
        ejemplos_texto += (
            f"  SMILES={ej['smiles']} | adduct={ej['adduct']} | "
            f"m/z={ej['mz']} | rings={extraer_caracteristicas(ej['smiles'])['num_rings']} | "
            f"CCS={ej['ccs']}\n"
        )

    prompt = f"""Predict the CCS (collision cross-section, in Ų) for this molecule in mass spectrometry.

Molecule:
  SMILES: {smiles}
  m/z: {mz:.4f}
  Adduct: {adduct_norm}
  Adduct effect: {info['effect']}
  Ionic charge: {info['charge']:+d} | Mass contribution of adduct: {info['mass_add']:.3f} Da
  Rings: {feat['num_rings']} | Branches: {feat['num_branches']} | Aromatic atoms: {feat['aromatic_atoms']}
  N={feat['num_N']} O={feat['num_O']} S={feat['num_S']} Charge={feat['net_charge']}

Rules:
- Larger m/z → larger CCS
- More rings/aromatic atoms → more compact → smaller CCS
- More flexible chains → larger CCS
- Adducts with larger ions (Na, K) → slightly larger CCS than [M+H]+
- Multiply charged ions → more compact geometry → smaller CCS relative to m/z
- Negative mode ([M-H]-) → typically slightly smaller CCS than positive mode

Reference molecules with known CCS (same or similar adduct preferred):
{ejemplos_texto}
Dataset CCS range: {stats['ccs_min']:.1f} - {stats['ccs_max']:.1f} Ų (average: {stats['ccs_avg']:.1f})

The adduct is {adduct} ({info['effect']}). Based on the reference SMILES, the reference molecules, the rules, and the adduct effect, the predicted CCS is:
"""
    #
    # the predicted CCS float value is (respond only with \\boxed{{number}}):
    return prompt


def construir_prompt_ejemplos(smiles, mz, adduct, stats, ejemplos, n = 5):
    feat = extraer_caracteristicas(smiles)

    # Información del aducto
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'})

    ejemplos_texto = ""
    for ej in ejemplos[:n]:
        ejemplos_texto += (
            f"  SMILES={ej['smiles']} | adduct={ej['adduct']} | "
            f"m/z={ej['mz']} | rings={extraer_caracteristicas(ej['smiles'])['num_rings']} | "
            f"CCS={ej['ccs']}\n"
        )

    prompt = f"""Predict the CCS (collision cross-section, in Ų) for this molecule in mass spectrometry.

Molecule:
  SMILES: {smiles}
  m/z: {mz:.4f}
  Adduct: {adduct_norm}
  Adduct effect: {info['effect']}
  Ionic charge: {info['charge']:+d} | Mass contribution of adduct: {info['mass_add']:.3f} Da
  Rings: {feat['num_rings']} | Branches: {feat['num_branches']} | Aromatic atoms: {feat['aromatic_atoms']}
  N={feat['num_N']} O={feat['num_O']} S={feat['num_S']} Charge={feat['net_charge']}

Reference molecules with known CCS (same or similar adduct preferred):
{ejemplos_texto}
Dataset CCS range: {stats['ccs_min']:.1f} - {stats['ccs_max']:.1f} Ų (average: {stats['ccs_avg']:.1f})

The adduct is {adduct} ({info['effect']}). Based on the reference SMILES, the reference molecules, and the adduct effect, the predicted CCS is:
"""
    return prompt


def construir_prompt_ce(smiles, mz, adduct):
    feat = extraer_caracteristicas(smiles)

    # Información del aducto
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'})

    prompt = f"""Predict the CCS (collision cross-section, in Ų) for this molecule in mass spectrometry.

Molecule:
  SMILES: {smiles}
  m/z: {mz:.4f}
  Adduct: {adduct_norm}
  Adduct effect: {info['effect']}
  Ionic charge: {info['charge']:+d} | Mass contribution of adduct: {info['mass_add']:.3f} Da
  Rings: {feat['num_rings']} | Branches: {feat['num_branches']} | Aromatic atoms: {feat['aromatic_atoms']}
  N={feat['num_N']} O={feat['num_O']} S={feat['num_S']} Charge={feat['net_charge']}

Rules:
- Larger m/z → larger CCS
- More rings/aromatic atoms → more compact → smaller CCS
- More flexible chains → larger CCS
- Adducts with larger ions (Na, K) → slightly larger CCS than [M+H]+
- Multiply charged ions → more compact geometry → smaller CCS relative to m/z
- Negative mode ([M-H]-) → typically slightly smaller CCS than positive mode

The adduct is {adduct} ({info['effect']}). Based on the reference SMILES, the rules, and the adduct effect, the predicted CCS is:
"""
    return prompt


def construir_prompt_simple(smiles, mz, adduct):
    feat = extraer_caracteristicas(smiles)

    # Información del aducto
    adduct_norm = normalizar_aducto(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'})

    prompt = f"""Predict the CCS (collision cross-section, in Ų) for this molecule in mass spectrometry.

Molecule:
  SMILES: {smiles}
  m/z: {mz:.4f}
  Adduct: {adduct_norm}
  Adduct effect: {info['effect']}
  Ionic charge: {info['charge']:+d} | Mass contribution of adduct: {info['mass_add']:.3f} Da
  Rings: {feat['num_rings']} | Branches: {feat['num_branches']} | Aromatic atoms: {feat['aromatic_atoms']}
  N={feat['num_N']} O={feat['num_O']} S={feat['num_S']} Charge={feat['net_charge']}

The adduct is {adduct} ({info['effect']}). Based on the reference SMILES and the adduct effect, the predicted CCS is:
"""
    return prompt


def predecir_ccs(model, tokenizer, prompt, mz_fallback, stats, max_new_tokens = 30):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    print(f"\t Tokens del prompt: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample = False, # mucho más rigido
            max_new_tokens = max_new_tokens,
            # use_cache = False, # Si me quedo sin VRAM
            pad_token_id = tokenizer.eos_token_id,
            repetition_penalty = 1.1
        )

    print(f"\t Tokens del output: {outputs.shape[1]}")

    respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_texto = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    respuesta = respuesta_completa[len(prompt_texto):].strip() # esto es solo la parte nueva

    print("\t Respuesta COMPLETA: " + json.dumps(respuesta_completa))
    print("\t Respuesta CRUDA: " + json.dumps(respuesta))

    # 1. Intentar parsear JSON
    try:
        match = re.search(r'\{[^{}]+\}', respuesta, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if "predicted_ccs" in data:
                ccs = float(data["predicted_ccs"])
                if ccs > 0:
                    return {
                        "predicted_ccs": round(ccs, 2),
                        "shape": data.get("shape", "Unknown"),
                        "reasoning": data.get("reasoning", "From JSON"),
                        "fallback": False
                    }
    except:
        pass

    # 2. Buscar números en la respuesta
    patrones_numero = [
        r'(?:is|=|:)\s*(\d{3}\.?\d*)',
        r'(\d{3}\.\d+)',
        r'(\d{3})',
    ]
    for patron in patrones_numero:
        matches = re.findall(patron, respuesta)
        for m in matches:
            ccs = float(m)
            if ccs > 0:
                return {
                    "predicted_ccs": round(ccs, 2),
                    "shape": "Unknown",
                    "reasoning": "Extracted from model text",
                    "fallback": True
                }

    # 3. Fallback heurístico
    ratio = (mz_fallback - stats['mz_min']) / max(stats['mz_max'] - stats['mz_min'], 1)
    ccs_estimado = stats['ccs_min'] + ratio * (stats['ccs_max'] - stats['ccs_min'])
    return {
        "predicted_ccs": round(ccs_estimado, 2),
        "shape": "Moderate",
        "reasoning": "Heuristic fallback based on m/z interpolation",
        "fallback": True
    }


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchao")
from torchao.quantization import quantize_, Int8WeightOnlyConfig

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
        dtype = torch.float32, # CPU requiere float32 para quantize_dynamic
        low_cpu_mem_usage = False
    )
    print(" - Modelo cargado correctamente")

    # Cuantización - Reducción de pesos a int8 para menor uso de memoria
    print("Cuantizando de modelo a int8...")
    quantize_(model, Int8WeightOnlyConfig())
    print(" - Modelo cuantizado correctamente")

    model.eval()  # Modo inferencia: desactiva dropout y gradientes
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
    ]

    prompts_func = {
        "simple": lambda s, m, a: construir_prompt_simple(s, m, a),
        "conocimiento experto": lambda s, m, a: construir_prompt_ce(s, m, a),
        "ejemplos": lambda s, m, a: construir_prompt_ejemplos(s, m, a, STATS, seleccionar_ejemplos(DATOS_TRAIN, s, m, a), n = 5),
        "completo": lambda s, m, a: construir_prompt_completo(s, m, a, STATS, seleccionar_ejemplos(DATOS_TRAIN, s, m, a), n = 5),
    }

    print("=" * 70)
    print("TEST DE PROMPTS")
    print("=" * 70)

    for i, caso in enumerate(test_cases, 1):
        smiles, mz, adduct = caso["smiles"], caso["mz"], caso["adduct"]
        print(f"\n{'='*70}")
        print(f"COMPUESTO {i} | m/z = {mz} | Aducto = {adduct}")
        print(f"SMILES: {smiles[:60]}...")
        print(f"{'='*70}")

        for nombre, fn_prompt in prompts_func.items():
            prompt = fn_prompt(smiles, mz, adduct)
            print(f" - Prompt con {nombre}: ")
            resultado = predecir_ccs(MODEL, TOKENIZER, prompt, mz, STATS)
            fallback_str = " [FALLBACK]" if resultado["fallback"] else ""
            print(f"\t CCS = {resultado['predicted_ccs']:.2f} Ų{fallback_str}")
            print(f"\t Reasoning: {resultado['reasoning'][:80]}")

    print(f"\n{'='*70}")
    print("TEST COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    if inicializar_app():
        test_prompts()
    else:
        print("ERROR en la inicialización. Verifica la configuración.")
