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
                    'adduct': row['Adduct'].strip() if 'Adduct' in row else '[M+H]+',
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


def seleccionar_ejemplos(datos, smiles_input, mz_input, adduct_input, n=5):
    caract_input = extraer_caracteristicas(smiles_input)
    similitudes = []
    for d in datos:
        caract = extraer_caracteristicas(d['smiles'])
        sim_mz = 1 / (1 + abs(d['mz'] - mz_input) / 100)
        sim_longitud = 1 / (1 + abs(caract['length'] - caract_input['length']) / 10)
        sim_aducto = 1.0 if d['adduct'] == adduct_input else 0.3
        sim_estructura = 1 / (1 + abs(caract['num_rings'] - caract_input['num_rings']))
        similitud = 0.4 * sim_mz + 0.2 * sim_longitud + 0.2 * sim_aducto + 0.2 * sim_estructura
        similitudes.append((similitud, d))
    similitudes.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in similitudes[:n]]


def buscar_en_dataset(smiles, adduct, dataset):
    for row in dataset:
        if row["smiles"] == smiles and row["adduct"] == adduct:
            return row["ccs"]
    return None


def construir_prompt_simple(smiles, mz, adduct, stats, ejemplos):
    feat = extraer_caracteristicas(smiles)
    ejemplos_texto = ""
    for ej in ejemplos[:4]:
        ejemplos_texto += f"  m/z={ej['mz']:.1f} | rings={extraer_caracteristicas(ej['smiles'])['num_rings']} | CCS={ej['ccs']:.2f}\n"

    prompt = f"""Predict the CCS (collision cross-section, in Ų) for this molecule in mass spectrometry.

Molecule:
  SMILES: {smiles}
  m/z: {mz:.4f}
  Adduct: {adduct}
  Rings: {feat['num_rings']} | Branches: {feat['num_branches']} | Aromatic atoms: {feat['aromatic_atoms']}
  N={feat['num_N']} O={feat['num_O']} S={feat['num_S']} Charge={feat['net_charge']}

Reference molecules with known CCS:
{ejemplos_texto}
Dataset CCS range: {stats['ccs_min']:.1f} - {stats['ccs_max']:.1f} Ų (average: {stats['ccs_avg']:.1f})

Rules: larger m/z → larger CCS. More rings/aromatic atoms → more compact → smaller CCS. More flexible chains → larger CCS.

Based on the reference molecules and the rules, the predicted CCS is:
"""
    return prompt


def predecir_ccs(model, tokenizer, prompt, mz_fallback, stats, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_texto = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    respuesta = respuesta_completa[len(prompt_texto):].strip()

    # Intentar parsear JSON
    try:
        match = re.search(r'\{[^{}]+\}', respuesta, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if "predicted_ccs" in data:
                ccs = float(data["predicted_ccs"])
                if stats['ccs_min'] <= ccs <= stats['ccs_max']:
                    return {
                        "predicted_ccs": round(ccs, 2),
                        "shape": data.get("shape", "Unknown"),
                        "reasoning": data.get("reasoning", "From JSON"),
                        "fallback": False
                    }
    except:
        pass

    # Buscar números en la respuesta
    patrones_numero = [
        r'(?:is|=|:)\s*(\d{3}\.?\d*)',
        r'(\d{3}\.\d+)',
        r'(\d{3})',
    ]

    for patron in patrones_numero:
        matches = re.findall(patron, respuesta)
        for m in matches:
            ccs = float(m)
            if stats['ccs_min'] <= ccs <= stats['ccs_max']:
                return {
                    "predicted_ccs": round(ccs, 2),
                    "shape": "Unknown",
                    "reasoning": "Extracted from model text",
                    "fallback": True
                }

    # Fallback heurístico
    ratio = (mz_fallback - stats['mz_min']) / max(stats['mz_max'] - stats['mz_min'], 1)
    ccs_estimado = stats['ccs_min'] + ratio * (stats['ccs_max'] - stats['ccs_min'])
    ccs_estimado = max(stats['ccs_min'], min(stats['ccs_max'], ccs_estimado))

    return {
        "predicted_ccs": round(ccs_estimado, 2),
        "shape": "Moderate",
        "reasoning": f"Heuristic fallback based on m/z interpolation",
        "fallback": True
    }


def cargar_modelo():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print("Modelo cargado correctamente")
    return model, tokenizer


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global MODEL, TOKENIZER, DATOS_TRAIN, STATS

    try:
        data = request.json
        smiles = data.get('smiles', '').strip()
        mz = float(data.get('mz', 0))
        adduct = data.get('adduct', '[M+H]+').strip()

        if not smiles or mz <= 0:
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

        # Seleccionar ejemplos similares
        ejemplos = seleccionar_ejemplos(DATOS_TRAIN, smiles, mz, adduct, n=5)

        # Construir prompt
        prompt = construir_prompt_simple(smiles, mz, adduct, STATS, ejemplos)

        # Predecir
        resultado = predecir_ccs(MODEL, TOKENIZER, prompt, mz, STATS)

        # Añadir información adicional
        resultado['molecular_features'] = extraer_caracteristicas(smiles)
        resultado['similar_compounds'] = [
            {'smiles': ej['smiles'], 'mz': ej['mz'], 'ccs': ej['ccs'], 'adduct': ej['adduct']}
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
    print(f" - CCS range: {STATS['ccs_min']:.1f} - {STATS['ccs_max']:.1f} Å²")
    print(f" - Correlación m/z-CCS: {STATS['correlacion_mz_ccs']:.3f}")

    # Cargar modelo
    print("Cargando modelo DeepSeek...")
    MODEL, TOKENIZER = cargar_modelo()
    print(" - Modelo cargado y listo")
    print(" - Aplicación lista para recibir peticiones")
    print("=" * 70)
    return True


if __name__ == '__main__':
    if inicializar_app():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("ERROR en la inicialización. Verifica la configuración.")
