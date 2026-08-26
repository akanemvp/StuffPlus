import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model", "artifacts")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles", "output")

# Statcast VAA/HAA constants
Y0 = 50.0       # measurement point, feet
YF = 17 / 12    # home plate width (17 inches) converted to feet

# Stuff+ normalization
STUFF_MEAN = 100
STUFF_STD = 10

# Model split
RANDOM_STATE = 42
TEST_SIZE = 0.2

# SQLite DB
DB_PATH = os.path.join(DATA_DIR, "statcast.db")

# All Statcast pitch type codes + display names
PITCH_TYPES = {
    # Fastballs
    "FF": "4-Seam Fastball",
    "FA": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",

    # Breaking balls
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CS": "Slow Curve",
    "SC": "Screwball",
    "GY": "Gyroball",

    # Offspeed
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Forkball",

    # Specialty
    "KN": "Knuckleball",
    "EP": "Eephus",
}

# Pitch types to EXCLUDE (non-pitch events)
EXCLUDE_PITCH_TYPES = {"PO", "IN", "AB", "NP", ""}

# Pitch family groupings — GMM clustering runs within each family.
# Sweeper (ST) is separate from SL so gyro/traditional/sweeper clusters
# emerge naturally within each family without cross-contamination.
PITCH_TYPE_MODELS = {
    "ff": ["FF", "FA"],
    "si": ["SI"],
    "fc": ["FC"],
    "sl": ["SL", "ST", "SV", "SC", "GY"],
    "cu": ["CU", "KC", "CS"],
    "ch": ["CH"],
    "fs": ["FS", "FO"],
}

# Alias used by cluster.py
PITCH_FAMILIES = PITCH_TYPE_MODELS

# Clustering features — physical characteristics only, NO arm_angle.
# Used to fit global GMM clusters across all pitch types. These features
# define WHAT a pitch is physically, not WHERE it's thrown from.
# spin_axis is the key discriminator (gyro slider ~180° clusters with curveballs).
CLUSTER_FEATURES = [
    "release_speed",  # separates fastball family from breaking/offspeed
    "ivb_accel",      # vertical movement force — separates rise from drop
    "hb_accel_arm",   # horizontal movement — gyro(≈0) vs sweeper(large neg)
    "spin_axis",      # key discriminator: gyro slider (179°) clusters with CU (180°)
]

# Reverse lookup: pitch_type code → model key
PITCH_TYPE_TO_MODEL = {
    pt: model_key
    for model_key, pts in PITCH_TYPE_MODELS.items()
    for pt in pts
}

# Run value column from Statcast
RV_COL = "delta_run_exp"

# Arm angle bins — used by movement baselines pipeline for ivb_adj/hb_adj regression
ARM_ANGLE_BINS = 6
MIN_PITCHES_BASELINE = 30

# Seasons
TRAINING_SEASONS = [2022, 2023, 2024, 2025]   # 2022-2025 window for the HR-only Method A model; 2025 is training data (not scored/displayed)
LIVE_SEASON = 2026
