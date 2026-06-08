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
TRAIN_DATA = None
STATS = None


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


ADDUCT_INFO = {
    '[M+H]':     {'charge': 1,  'mass_add': 1.007,  'effect': 'protonated, baseline reference'},
    '[M+Na]':    {'charge': 1,  'mass_add': 22.989, 'effect': 'sodium adduct, ~3-5 Å² larger than [M+H]+'},
    '[M-H]':     {'charge': -1, 'mass_add': -1.007, 'effect': 'deprotonated, ~2-5 Å² smaller than [M+H]+'},
}

# Builds a simple prompt using only the target compound information.
def build_prompt(smiles, adduct):
    adduct_norm = normalize_adduct(adduct)
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
            result = accept(value, 'final_css_tag')
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
def classify_prediction(ccs_pred, stats):

    avg = round(stats['ccs_avg'])
    if abs(ccs_pred - avg) <= 0.5:
        return 'dataset_avg', False

    return 'interpolated', True

# IF RE-RUNNING THE TESTS, CONSIDER CHANGING:
#  - top_p = 0.95 (official recommendation)
#  - max_new_tokens = 20000 (two fallbacks observed with the 1.5B model)
#  - Investigate the temperature value and understand why 0.3 was used instead of the recommended range 0.5-0.7
def predict_ccs(model, tokenizer, prompt, stats):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=5000)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample = False,           # 1.5B -> False ; Bigger models (7B) -> True
            temperature = None,          # 1.5B -> None  ; Bigger models (7B) -> 0.3
            top_p = None,                # 1.5B -> None  ; Bigger models (7B) -> 0.9
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
    pred_type, is_valid = classify_prediction(result['predicted_ccs'], stats)
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


def initialize_app():
    global MODEL, TOKENIZER, TRAIN_DATA, STATS

    print("=" * 70)
    print("Initializing CCS prediction application.")
    print("=" * 70)

    # Load data
    csv_path = r"../data/processed/other/train.csv"
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
    print(f" - CCS range: {STATS['ccs_min']:.1f} - {STATS['ccs_max']:.1f} Å²")

    # Load model
    print("Loading DeepSeek model...")
    MODEL, TOKENIZER = load_model()
    print(" - Model loaded and ready")
    print(" - Application ready to receive requests")
    print("=" * 70)
    return True


def test():

    global MODEL, TOKENIZER, TRAIN_DATA, STATS

    # 10 test cases extracted from train.csv
    test_cases = [
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
    print(f"EFFICIENCY TEST\nTest start: {datetime.now()}")

    for i, case in enumerate(test_cases, 1):
        smiles, adduct = case["smiles"], case["adduct"]
        print(f"{'='*70}")
        print(f"COMPOUND {i} | Adduct = {adduct}")
        print(f"SMILES: {smiles}")
        print(f"{'='*70}")

        # Clear cache between predictions
        torch.cuda.empty_cache()  # just in case, even on CPU
        if hasattr(MODEL, 'reset_cache'):
            MODEL.reset_cache()

        prompt = build_prompt(smiles, adduct)
        result = predict_ccs(MODEL, TOKENIZER, prompt, STATS)

        fallback_str = " [FALLBACK]" if result["fallback"] else ""
        print(f" CCS = {result['predicted_ccs']:.2f} Å²{fallback_str}")
        print(f" Reasoning: {result['reasoning']}")

    print(f"\n{'='*70}")
    print(f"TEST COMPLETE!\nTest end: {datetime.now()}")
    print("=" * 70)

if __name__ == '__main__':
    if initialize_app():
        test()
    else:
        print("ERROR during initialization. Check the configuration.")
