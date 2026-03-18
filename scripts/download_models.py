import os
import sys
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

DIRECTORIO_BASE = r"D:\Modelos TFG"
MODELOS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
]
# Token de HuggingFace (necesario solo para modelos privados o con licencia)
HF_TOKEN = None # Se deja en None si el modelo es público

# INSTALACIÓN DE DEPENDENCIAS
def instalar_si_falta(paquete):
    import importlib
    try:
        importlib.import_module(paquete.replace("-", "_"))
    except ImportError:
        print(f"  Instalando {paquete}...")
        os.system(f"{sys.executable} -m pip install {paquete} -q")

print("Verificando dependencias...")
for pkg in ["huggingface_hub", "tqdm"]:
    instalar_si_falta(pkg)


def obtener_nombre_carpeta(repo_id: str) -> str:
    """Convierte 'org/nombre-modelo' → 'nombre-modelo'"""
    return repo_id.split("/")[-1]

def tamanio_carpeta(ruta: str) -> str:
    """Devuelve el tamaño total de una carpeta en formato legible."""
    total = 0
    for dirpath, _, filenames in os.walk(ruta):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    for unidad in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024:
            return f"{total:.1f} {unidad}"
        total /= 1024
    return f"{total:.1f} PB"


def descargar_modelo(repo_id: str, directorio_base: str, token=None):
    nombre = obtener_nombre_carpeta(repo_id)
    destino = os.path.join(directorio_base, nombre)

    print(f"\n{'='*60}")
    print(f"  Modelo : {repo_id}")
    print(f"  Destino: {destino}")
    print(f"{'='*60}")

    os.makedirs(destino, exist_ok=True)

    # Comprobar si ya existe y tiene contenido
    archivos_existentes = [
        f for f in os.listdir(destino)
        if os.path.isfile(os.path.join(destino, f))
    ]
    if archivos_existentes:
        print(f"  La carpeta ya contiene {len(archivos_existentes)} archivo(s).")
        respuesta = input("  ¿Descargar de nuevo / completar archivos que falten? [s/N]: ").strip().lower()
        if respuesta != "s":
            print("  Omitido.")
            return

    try:
        print("  Iniciando descarga (puede tardar varios minutos)...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=destino,
            token=token,
            local_dir_use_symlinks=False,   # Copia real, sin symlinks
            resume_download=True,           # Reanuda si se interrumpe
            ignore_patterns=[               # Excluye archivos innecesarios
                "*.msgpack",
                "flax_model*",
                "tf_model*",
                "rust_model*",
                "onnx/*",
            ],
        )
        print(f"\n  ✓ Descarga completada.")
        print(f"  Tamaño en disco: {tamanio_carpeta(destino)}")

    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print(f"\n  ✗ Error de autenticación.")
            print("    Este modelo requiere un token de HuggingFace.")
            print("    1. Ve a https://huggingface.co/settings/tokens")
            print("    2. Crea un token de lectura")
            print("    3. Añádelo en HF_TOKEN al inicio de este script")
        else:
            print(f"\n  ✗ Error HTTP: {e}")

    except Exception as e:
        print(f"\n  ✗ Error inesperado: {e}")


def verificar_modelo(ruta: str) -> dict:
    """Comprueba que los archivos esenciales están presentes."""
    archivos = os.listdir(ruta) if os.path.exists(ruta) else []
    return {
        "config.json":            "config.json" in archivos,
        "tokenizer_config.json":  "tokenizer_config.json" in archivos,
        "tokenizer.json":         "tokenizer.json" in archivos,
        "tokenizer.model":        "tokenizer.model" in archivos,
        "pesos (.safetensors)":   any(f.endswith(".safetensors") for f in archivos),
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Descargador de modelos HuggingFace")
    print("="*60)
    print(f"  Directorio base: {DIRECTORIO_BASE}")
    print(f"  Modelos a descargar: {len(MODELOS)}")

    os.makedirs(DIRECTORIO_BASE, exist_ok=True)

    for repo_id in MODELOS:
        descargar_modelo(repo_id, DIRECTORIO_BASE, token=HF_TOKEN)

    # ── Resumen final ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESUMEN DE MODELOS DESCARGADOS")
    print("="*60)

    for repo_id in MODELOS:
        nombre = obtener_nombre_carpeta(repo_id)
        ruta = os.path.join(DIRECTORIO_BASE, nombre)
        checks = verificar_modelo(ruta)

        print(f"\n  {nombre}")
        for archivo, ok in checks.items():
            estado = "✓" if ok else "✗ FALTA"
            print(f"    {estado}  {archivo}")

        if os.path.exists(ruta):
            print(f"    Tamaño total: {tamanio_carpeta(ruta)}")

    print("\n  Proceso finalizado.")