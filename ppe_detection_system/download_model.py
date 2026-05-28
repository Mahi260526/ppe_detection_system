"""
Script to download PPE Detection YOLOv8 model (Helmet, Vest, Mask).
Primary: Construction-PPE-Detection (Ansarimajid) – ppe.pt, Helmet/Vest/Mask + Person.
Fallback: Hugging Face Hansung-Cho model.
"""
import os
import shutil
import requests
from pathlib import Path

# Construction-PPE-Detection: helmet, vest, mask, person (https://github.com/Ansarimajid/Construction-PPE-Detection)
CONSTRUCTION_PPE_URL = "https://raw.githubusercontent.com/Ansarimajid/Construction-PPE-Detection/main/Model/ppe.pt"
# Fallback
MODEL_REPO = "Hansung-Cho/yolov8-ppe-detection"
MODEL_FILENAME = "best.pt"

def download_file(url, destination):
    """Download a file from URL to destination"""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    print(f"\nDownloaded to {destination}")
    return True

def download_from_huggingface():
    """Download best.pt from Hugging Face Hub (Helmet, Vest, Mask/face protection)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    return path

def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "best.pt"

    print("PPE Detection YOLOv8 – Helmet, Vest, Mask (Construction-PPE-Detection)")
    print("Source: https://github.com/Ansarimajid/Construction-PPE-Detection\n")

    # 1) Prefer Construction-PPE-Detection model (Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Mask, NO-Mask, Person)
    try:
        download_file(CONSTRUCTION_PPE_URL, str(model_path))
        print(f"Successfully saved model to {model_path}")
        print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        return
    except Exception as e:
        print(f"Construction-PPE download failed: {e}")

    # 2) Fallback: Hugging Face
    try:
        downloaded = download_from_huggingface()
        shutil.copy(downloaded, model_path)
        print(f"Successfully saved model to {model_path}")
        print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        return
    except Exception as e:
        print(f"Hugging Face download failed: {e}")

    # 3) Direct HF URL
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}"
    try:
        download_file(url, str(model_path))
        print(f"Successfully downloaded model to {model_path}")
        return
    except Exception as e:
        print(f"Direct URL failed: {e}")

    print("\nManual: download ppe.pt from https://github.com/Ansarimajid/Construction-PPE-Detection (Model/ppe.pt) and save as models/best.pt")

if __name__ == "__main__":
    main()

