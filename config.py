import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model", "artifacts")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles", "output")

Y0 = 50.0
YF = 17 / 12


DB_PATH = os.path.join(DATA_DIR, "statcast.db")

PITCH_TYPES = {
    "FF": "4-Seam Fastball",
    "FA": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",

    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CS": "Slow Curve",
    "SC": "Screwball",
    "GY": "Gyroball",

    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Forkball",

    "KN": "Knuckleball",
    "EP": "Eephus",
}

EXCLUDE_PITCH_TYPES = {"PO", "IN", "AB", "NP", ""}

PITCH_TYPE_MODELS = {
    "ff": ["FF", "FA"],
    "si": ["SI"],
    "fc": ["FC"],
    "sl": ["SL", "ST", "SV", "SC", "GY"],
    "cu": ["CU", "KC", "CS"],
    "ch": ["CH"],
    "fs": ["FS", "FO"],
}


ARM_ANGLE_BINS = 6
MIN_PITCHES_BASELINE = 30

TRAINING_SEASONS = [2022, 2023, 2024, 2025]
