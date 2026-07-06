# config.py
"""
Global configuration constants, file paths, and lookup tables for the FIFA World Cup 2026
knockout stage xG & probability generator.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Directory & file paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../Training Data")) + "/"
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data")) + "/"

# NEW: Cache directory inside the ai_brain folder
CACHE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "cache")) + "/" 

# Ensure both directories exist before running
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Rerouted Files (Models and Pickles -> CACHE_DIR)
# ---------------------------------------------------------------------------
MODEL_PATH = f"{CACHE_DIR}best_model.pt"
TEAM_MAP_FILE = f"{CACHE_DIR}team_mapping.pkl"
ELO_CACHE_FILE = f"{CACHE_DIR}elo_state_cache.pkl"
SQUAD_CACHE_FILE = f"{CACHE_DIR}squad_cache.pkl"

# ---------------------------------------------------------------------------
# Output Files (CSVs and TXTs -> OUTPUT_DIR)
# ---------------------------------------------------------------------------
DEBUG_FILE = f"{OUTPUT_DIR}team_features_debug.txt"
ALIVE_FILE = f"{OUTPUT_DIR}alive_teams.txt"
TRANSFERMARKT_FILE = f"{DATA_DIR}transfermarkt_squad_values_2026.csv"

# ---------------------------------------------------------------------------
# Run‑time flags
# ---------------------------------------------------------------------------
FORCE_FRESH = True                 # Delete caches and retrain with current features

# ---------------------------------------------------------------------------
# Modelling & training hyperparameters
# ---------------------------------------------------------------------------
HISTORY_LEN = 10
EMBEDDING_DIM = 16
MIN_YEAR = 2000
CURRENT_YEAR = 2026
REF_DATE = pd.Timestamp('2026-06-01')
TEMPORAL_DECAY_LAMBDA = 0.15

EPOCHS = 100
LEARNING_RATE = 0.0005
BATCH_SIZE = 256

N_JOBS = -1
NUM_CORES = os.cpu_count()  # will be computed at runtime, ok here

# ---------------------------------------------------------------------------
# Elo parameters
# ---------------------------------------------------------------------------
ELO_K = 32                        # base K‑factor

# ---------------------------------------------------------------------------
# Poisson Monte Carlo simulations
# ---------------------------------------------------------------------------
POISSON_SIMULATIONS = 50000
POISSON_RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Name mapping (standardise team names across data sources)
# ---------------------------------------------------------------------------
name_map = {
    'USA': 'United States',
    'IR Iran': 'Iran',
    'Korea Republic': 'South Korea',
    'Congo DR': 'DR Congo',
    'Curacao': 'Curaçao',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Côte d\'Ivoire': 'Ivory Coast',
    'Korea DPR': 'North Korea',
    'RCS': 'Czech Republic',
    'Zaire': 'DR Congo',
    'Yugoslavia': 'Serbia',
    'Netherlands Antilles': 'Curaçao',
    'Türkiye': 'Turkey',
    'Korea, South': 'South Korea',
    'Cote d\'Ivoire': 'Ivory Coast',
    'Ivory Coast': 'Ivory Coast',
    'Iran, Islamic Republic of': 'Iran',
}

# ---------------------------------------------------------------------------
# 2026 World Cup group stage composition
# ---------------------------------------------------------------------------
GROUPS = {
    'A': ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    'B': ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    'C': ["Brazil", "Morocco", "Haiti", "Scotland"],
    'D': ["United States", "Paraguay", "Australia", "Turkey"],
    'E': ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    'F': ["Netherlands", "Japan", "Sweden", "Tunisia"],
    'G': ["Belgium", "Egypt", "Iran", "New Zealand"],
    'H': ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    'I': ["France", "Senegal", "Iraq", "Norway"],
    'J': ["Argentina", "Algeria", "Austria", "Jordan"],
    'K': ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    'L': ["England", "Croatia", "Ghana", "Panama"]
}

# Build lookup: team -> group letter
team_to_group = {team: grp for grp, teams in GROUPS.items() for team in teams}

# Sorted list of all participating teams
ALL_TEAMS = sorted(team_to_group.keys())

# ---------------------------------------------------------------------------
# Feature names & asymmetric dropout probabilities (7 static features)
# ---------------------------------------------------------------------------
# Add the number of static features
NUM_STATIC_FEATURES = 11

# Replace the old FEATURE_NAMES list
FEATURE_NAMES = [
    "Elo Diff",
    "Squad Sum Diff",
    "Squad Median Diff",
    "Squad Var Diff",
    "Count >50M Diff",
    "FIFA Diff",
    "Weighted Margin Diff",
    "Age Avg Diff",
    "Age Var Diff",
    "DaysSinceLastMatch Diff",
    "ValuePerCapRatio Diff",
]

# Extend MASK_PROBS – drop Elo with 0.5, others with 0.1
MASK_PROBS = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]




