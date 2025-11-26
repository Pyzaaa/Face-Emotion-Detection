import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_affectnet():
    raw_dir = "data/affectnet_raw"
    yolo_dir = os.path.join(raw_dir, "YOLO_format")
    zip_path = "data/affectnet-yolo-format.zip"

    if os.path.exists(yolo_dir):
        print("[OK] Dataset już pobrany:", yolo_dir)
        return yolo_dir

    print("[INFO] Pobieranie AffectNet z Kaggle...")
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files("fatihkgg/affectnet-yolo-format",
                               path="data",
                               unzip=False)
    print("[INFO] Rozpakowywanie...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
    os.remove(zip_path)
    print("[OK] Dataset pobrany i rozpakowany:", yolo_dir)
    return yolo_dir

download_affectnet()