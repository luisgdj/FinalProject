import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, render_template, request, jsonify
import json
import re
import csv

app = Flask(__name__)

MODEL = None
TOKENIZER = None
TRAIN_DATA = None
STATS = None

def extract_features(smiles):
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


def read_csv(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    'smiles': row['smiles'].strip(),
                    'adduct': row['Adduct'].strip(),
                    'ccs': float(row['CCS_AVG'])
                })
            except (ValueError, KeyError):
                continue
    return data


def analyze_data(data):
    ccs_values = [d['ccs'] for d in data]
    stats = {
        'ccs_min': min(ccs_values),
        'ccs_max': max(ccs_values),
        'ccs_avg': sum(ccs_values) / len(ccs_values)
    }
    return stats


def normalize_adduct(adduct):
    # Strips the trailing charge character to unify adduct formats.
    return adduct.rstrip('+-').strip()

# Selects N structurally similar molecules from the dataset for the given input.
def select_examples(data, input_smiles, input_adduct, n):

    # Step 1: Compute descriptors for the input compound.
    input_features = extract_features(input_smiles)
    adduct_norm = normalize_adduct(input_adduct)

    # Step 2: Filter the training set by adduct.
    filtered_data = [d for d in data if normalize_adduct(d['adduct']) == adduct_norm]
    # Fallback: if not enough examples for the same adduct, use the full dataset.
    if len(filtered_data) < n:
        print(f"Warning: only {len(filtered_data)} examples for adduct {adduct_norm}, using full dataset")
        filtered_data = data

    similarities = []
    for d in filtered_data:
        features = extract_features(d['smiles'])

        # Step 3: Compute a similarity score for each structural descriptor.
        #  - For each molecule in the filtered set
        #  - Compare its descriptors against those of the input compound
        #  - Produce a score between 0 and 1
        sim_length   = 1 / (1 + abs(features['length']         - input_features['length']) / 10)
        sim_rings    = 1 / (1 + abs(features['num_rings']      - input_features['num_rings']))
        sim_aromatic = 1 / (1 + abs(features['aromatic_atoms'] - input_features['aromatic_atoms']) / 3)

        # Step 4: Combine the three scores with weights (structure > length > aromaticity).
        similarity = 0.50 * sim_rings + 0.30 * sim_length + 0.20 * sim_aromatic
        similarities.append((similarity, d))

    # Step 5: Sort by score and return the top n.
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in similarities[:n]]


def search_dataset(smiles, adduct, dataset):
    adduct_norm = normalize_adduct(adduct)
    for row in dataset:
        if row["smiles"] == smiles and normalize_adduct(row["Adduct"]) == adduct_norm:
            return row["ccs"]
    return None


ADDUCT_INFO = {
    '[M+H]':     {'charge': 1,  'mass_add': 1.007,  'effect': 'protonated, baseline reference'},
    '[M+Na]':    {'charge': 1,  'mass_add': 22.989, 'effect': 'sodium adduct, ~3-5 Å² larger than [M+H]+'},
    '[M-H]':     {'charge': -1, 'mass_add': -1.007, 'effect': 'deprotonated, ~2-5 Å² smaller than [M+H]+'},
}

# Builds a prompt using expert knowledge, database examples, and the target compound information.
def build_prompt(smiles, adduct, TRAIN_DATA, examples):
    adduct_norm = normalize_adduct(adduct)
    info = ADDUCT_INFO.get(adduct_norm, {
        'charge': 1, 'mass_add': 0, 'effect': 'unknown adduct type'
    })

    # Build the examples block (same format as the target: SMILES + Adduct + CCS)
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        ex_adduct_norm = normalize_adduct(ex['adduct'])
        examples_text += (
            f"  Example {i}:\n"
            f"    SMILES: {ex['smiles']}\n"
            f"    Adduct: {ex_adduct_norm}\n"
            f"    CCS: {ex['ccs']:.2f} Å²\n\n"
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

Below are {len(examples)} reference molecules with experimentally measured CCS values, selected for their structural similarity to the target. Use them to identify patterns and infer the CCS of the target molecule.

Reference molecules:

{examples_text}To estimate CCS, follow these steps:
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


# Extracts the predicted CCS from the model's raw response.
def parse_response(raw_response):

    # Initial cleanup
    # Remove markdown bold/italic that interferes with regex
    text = raw_response.replace('**', '').replace('*', '')
    # Remove common LaTeX artifacts: $...$, $$...$$, \[ \], \( \)
    text = re.sub(r'\$+', '', text)
    text = re.sub(r'\\[\[\]\(\)]', '', text)
    text = text.strip()

    # Check the <think> block
    if '<think>' in text and '</think>' not in text:
        return {'predicted_ccs': None, 'fallback': True, 'source': 'think_unclosed'}
    if '</think>' not in text:
        return {'predicted_ccs': None, 'fallback': True, 'source': 'no_think_tag'}

    # From here on, work only with the post-think section
    post_text = text.split('</think>')[-1].strip()

    def accept(value, source):
        if 50 <= value <= 500:  # reasonable CCS range
            return {'predicted_ccs': round(value, 2), 'fallback': False, 'source': source}
        return None

    NUM = r'([0-9]+(?:\.[0-9]+)?)'

    # STRATEGY 1: "Final CCS: <number>" in variants (with/without \boxed{})
    final_patterns = [
        rf'Final\s+CCS\s*[:=]?\s*\\?boxed\{{?\s*{NUM}\s*\}}?',  # "Final CCS: \boxed{200}" or "Final CCS: 200"
        rf'Final\s+CCS\s*[:=]\s*{NUM}',                          # "Final CCS: 200" or "Final CCS = 200"
        rf'Final\s+CCS\s+{NUM}'                                   # "Final CCS 200" (no separator)
    ]
    for pattern in final_patterns:
        match = re.search(pattern, post_text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            result = accept(value, 'final_ccs_tag')
            if result:
                return result

    # STRATEGY 2: \boxed{<number>} without a preceding "Final CCS:"
    match = re.search(rf'\\?boxed\{{?\s*{NUM}\s*\}}?', post_text)
    if match:
        value = float(match.group(1))
        result = accept(value, 'boxed_tag')
        if result:
            return result

    # STRATEGY 3: last plausible number in the CCS range
    matches = re.findall(r'\b([0-9]{2,3}(?:\.[0-9]+)?)\b', post_text)
    plausible = [float(m) for m in matches if 50 <= float(m) <= 500]
    if plausible:
        return {
            'predicted_ccs': round(plausible[-1], 2),
            'fallback': False,
            'source': 'last_plausible',
        }

    return {'predicted_ccs': None, 'fallback': True, 'source': 'no_number_found'}

# Classifies whether the prediction is valid or a degenerate value.
def classify_prediction(ccs_pred, examples, stats):

    reference_ccs = {round(ex['ccs'], 2) for ex in examples}
    avg = round(stats['ccs_avg'])

    if abs(ccs_pred - avg) <= 0.5:
        return 'dataset_avg', False

    if any(abs(ccs_pred - ref) <= 0.02 for ref in reference_ccs):
        return 'exact_copy', False

    return 'interpolated', True


def predict_ccs(model, tokenizer, prompt, stats, examples):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=5000)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample = False,           # 1.5B -> False ; 7B/14B -> True
            temperature = None,          # 1.5B -> None  ; 7B/14B -> 0.3
            top_p = None,                # 1.5B -> None  ; 7B/14B -> 0.9
            repetition_penalty = 1.15,   # Penalizes repeated tokens to prevent looping during long reasoning traces.
            max_new_tokens = 10000,      # Extra margin for the <think> block
            pad_token_id = tokenizer.eos_token_id
        )

    n_prompt_tokens = inputs['input_ids'].shape[1]
    n_total_tokens = outputs.shape[1]
    n_generated_tokens = n_total_tokens - n_prompt_tokens

    print(f" - Prompt tokens: {n_prompt_tokens}")
    print(f" - Total output tokens: {n_total_tokens}")
    print(f" - Generated tokens: {n_generated_tokens}")

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    response = full_response[len(prompt_text):].strip()

    print(" FULL RESPONSE: " + json.dumps(full_response))
    print(" RAW RESPONSE: " + json.dumps(response))

    result = parse_response(response)

    # Fallback on parse failure
    if result['fallback']:
        result['predicted_ccs'] = 0.0
        result['pred_type'] = 'heuristic_fallback'
        result['reasoning'] = "Heuristic fallback: parser found no valid number"
        return result

    # Classify the prediction
    pred_type, is_valid = classify_prediction(result['predicted_ccs'], examples, stats)
    result['predicted_ccs_raw'] = result['predicted_ccs']
    result['pred_type'] = pred_type

    if is_valid:
        result['reasoning'] = f"Model interpolation ({result['predicted_ccs']})"
    else:
        result['pred_type'] = pred_type  # 'exact_copy' or 'dataset_avg', no heuristic fallback

        if pred_type == 'exact_copy':
            result['reasoning'] = f"Model output accepted (reference copy: {result['predicted_ccs_raw']})"
        else:
            result['reasoning'] = f"Model output accepted (dataset avg: {result['predicted_ccs_raw']})"

    return result


def load_model():

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model_path = r"D:\Modelos TFG\DeepSeek-R1-Distill-Qwen-1.5B"  # Local path
    print(f" - Path: {model_path}")

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

    model.eval()  # Inference mode: disables dropout and gradients

    if hasattr(torch, 'compile'):
        print(" - Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
        print(" - Compilation complete")

    return model, tokenizer



@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global MODEL, TOKENIZER, TRAIN_DATA, STATS

    try:
        data = request.json
        smiles = data.get('smiles', '').strip()
        adduct = data.get('adduct', '[M+H]+').strip()

        if not smiles:
            return jsonify({'error': 'Invalid input parameters'}), 400

        # Check if the compound exists in the dataset
        real_ccs = search_dataset(smiles, adduct, TRAIN_DATA)
        if real_ccs is not None:
            return jsonify({
                'success': True,
                'predicted_ccs': round(real_ccs, 2),
                'shape': 'Known',
                'reasoning': 'Exact match found in training dataset',
                'fallback': False,
                'from_dataset': True,
                'molecular_features': extract_features(smiles)
            })

        # Select n structurally similar molecules from the dataset
        examples = select_examples(TRAIN_DATA, smiles, adduct, n=5)

        # Build prompt
        prompt = build_prompt(smiles, adduct, TRAIN_DATA, examples)

        # Predict
        result = predict_ccs(MODEL, TOKENIZER, prompt, STATS, examples)

        # Add additional information
        result['molecular_features'] = extract_features(smiles)
        result['similar_compounds'] = [
            {'smiles': ex['smiles'], 'ccs': ex['ccs'], 'adduct': ex['adduct']}
            for ex in examples[:3]
        ]
        result['success'] = True
        result['from_dataset'] = False

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/status')
def status():
    return jsonify({
        'model_loaded': MODEL is not None,
        'dataset_loaded': TRAIN_DATA is not None,
        'dataset_size': len(TRAIN_DATA) if TRAIN_DATA else 0,
        'stats': STATS if STATS else {}
    })


def initialize_app():
    global MODEL, TOKENIZER, TRAIN_DATA, STATS

    print("=" * 70)
    print("Initializing CCS prediction application.")
    print("=" * 70)

    # Load data
    csv_path = r"data/processed/train.csv"
    if not os.path.exists(csv_path):
        print(f"WARNING: File {csv_path} not found")
        print("Please make sure train.csv is at the correct path")
        return False

    print(f"Loading dataset: {csv_path}")
    TRAIN_DATA = read_csv(csv_path)
    print(f" - Dataset loaded: {len(TRAIN_DATA)} compounds")

    # Compute statistics
    print("Analyzing dataset statistics...")
    STATS = analyze_data(TRAIN_DATA)
    print(f" - CCS range: {STATS['ccs_min']:.2f} - {STATS['ccs_max']:.2f} Å²")

    # Load model
    print("Loading DeepSeek model...")
    MODEL, TOKENIZER = load_model()
    print(" - Model loaded and ready")
    print(" - Application ready to receive requests")
    print("=" * 70)
    return True


if __name__ == '__main__':
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    if initialize_app():
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        print("ERROR during initialization. Check the configuration.")
