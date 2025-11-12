import joblib  # type: ignore
import pandas as pd  # type: ignore
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "flood_model.pkl")


model = joblib.load(MODEL_PATH)

print("model loaded successfully")
