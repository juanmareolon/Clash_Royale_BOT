import os
import zipfile
import io
import requests

# === CONFIGURACIÓN ===
DATASET_URL = "https://universe.roboflow.com/clashroyaleenemydetector/cr-card-detection-8000-img-57ki6/dataset/1/download/yolov8" 

# Carpeta donde se va a descomprimir
TARGET_DIR = "BOT_CLASH_ROYALE/assets/dataset"

# ======================

def download_and_extract(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "dataset.zip")

    print(f"Descargando dataset desde:\n{url}\n")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB
    downloaded = 0

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                done = int(50 * downloaded / total_size) if total_size else 0
                print(f"\r[{'█' * done}{'.' * (50 - done)}] {downloaded/1e6:.1f} MB", end='')

    print("\nDescarga completa. Extrayendo archivos...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)

    os.remove(zip_path)
    print(f"✅ Dataset extraído en: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    download_and_extract(DATASET_URL, TARGET_DIR)
