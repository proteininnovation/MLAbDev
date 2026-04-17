# config.py
import os

# Central paths — used by ALL modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "build", "pretrained_models")
DB_DIR = os.path.join(BASE_DIR,"build","db")
PREDICTION_DIR = os.path.join(BASE_DIR, "build", "predictions")
TMP_DIR = os.path.join(BASE_DIR, "build", "tmp")
LOG = os.path.join(BASE_DIR, "build", "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LOG, exist_ok=True)
