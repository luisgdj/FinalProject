import os
import sys
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

# Change this path to where the models will be downloaded
BASE_DIRECTORY = r"D:\Modelos TFG"

# Add HuggingFace model names to download
MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
]
# HuggingFace token (only required for private or gated models)
HF_TOKEN = None  # Leave as None if the model is public


# DEPENDENCY INSTALLATION
def install_if_missing(package):
    import importlib
    try:
        importlib.import_module(package.replace("-", "_"))
    except ImportError:
        print(f"  Installing {package}...")
        os.system(f"{sys.executable} -m pip install {package} -q")

print("Checking dependencies...")
for pkg in ["huggingface_hub", "tqdm"]:
    install_if_missing(pkg)


# FUNCTIONS
def get_folder_name(repo_id: str) -> str:
    """Converts 'org/model-name' → 'model-name'"""
    return repo_id.split("/")[-1]

def folder_size(path: str) -> str:
    """Returns the total size of a folder in human-readable format."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} PB"


def download_model(repo_id: str, base_directory: str, token=None):
    name = get_folder_name(repo_id)
    destination = os.path.join(base_directory, name)

    print(f"\n{'='*60}")
    print(f"  Model  : {repo_id}")
    print(f"  Target : {destination}")
    print(f"{'='*60}")

    os.makedirs(destination, exist_ok=True)

    # Check if folder already exists and has content
    existing_files = [
        f for f in os.listdir(destination)
        if os.path.isfile(os.path.join(destination, f))
    ]
    if existing_files:
        print(f"  The folder already contains {len(existing_files)} file(s).")
        answer = input("  Download again / complete missing files? [y/N]: ").strip().lower()
        if answer != "y":
            print("  Skipped.")
            return

    try:
        print("  Starting download (this may take several minutes)...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=destination,
            token=token,
            local_dir_use_symlinks=False,   # Real copy, no symlinks
            resume_download=True,           # Resume if interrupted
            ignore_patterns=[               # Exclude unnecessary files
                "*.msgpack",
                "flax_model*",
                "tf_model*",
                "rust_model*",
                "onnx/*",
            ],
        )
        print(f"\n  Download complete.")
        print(f"  Size on disk: {folder_size(destination)}")

    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print(f"\nAuthentication error.")
            print("  This model requires a HuggingFace token.")
            print("   1. Go to https://huggingface.co/settings/tokens")
            print("   2. Create a read token")
            print("   3. Add it to HF_TOKEN at the top of this script")
        else:
            print(f"\n  HTTP error: {e}")

    except Exception as e:
        print(f"\n  Unexpected error: {e}")


def verify_model(path: str) -> dict:
    """Checks that essential files are present."""
    files = os.listdir(path) if os.path.exists(path) else []
    return {
        "config.json":            "config.json" in files,
        "tokenizer_config.json":  "tokenizer_config.json" in files,
        "tokenizer.json":         "tokenizer.json" in files,
        "tokenizer.model":        "tokenizer.model" in files,
        "weights (.safetensors)": any(f.endswith(".safetensors") for f in files),
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  HuggingFace Model Downloader")
    print("="*60)
    print(f"  Base directory : {BASE_DIRECTORY}")
    print(f"  Models to download: {len(MODELS)}")

    os.makedirs(BASE_DIRECTORY, exist_ok=True)

    for repo_id in MODELS:
        download_model(repo_id, BASE_DIRECTORY, token=HF_TOKEN)

    # Final summary
    print("\n" + "="*60)
    print("  DOWNLOADED MODELS SUMMARY")
    print("="*60)

    for repo_id in MODELS:
        name = get_folder_name(repo_id)
        path = os.path.join(BASE_DIRECTORY, name)
        checks = verify_model(path)

        print(f"\n  {name}")
        for file, ok in checks.items():
            status = "✓" if ok else "✗ MISSING"
            print(f"    {status}  {file}")

        if os.path.exists(path):
            print(f"    Total size: {folder_size(path)}")

    print("\n  Process complete.")
