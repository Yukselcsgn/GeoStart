# geo_dataset/config.py

from pathlib import Path

# Proje kök dizininden konumlar
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "images"
CSV_PATH = BASE_DIR / "metadata.csv"

# Dataloader ayarları
BATCH_SIZE = 32
NUM_WORKERS = 4
SHUFFLE = True
