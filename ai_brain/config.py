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
FORCE_FRESH = True                # Delete caches and retrain with current features

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

NUM_STATIC_FEATURES = 8

# New feature names (only 8)
FEATURE_NAMES = [
    "Elo Diff",
    "Squad Sum Diff",
    "Squad Median Diff",
    "Squad Var Diff",
    "Count >50M Diff",
    "Weighted Margin Diff",
    "Age Avg Diff",
    "ValuePerCapRatio Diff",
]

# Dropout probabilities (8 entries)
MASK_PROBS = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]



# Tournament → K weight (World Football Elo system)
K_WEIGHTS = {
    # Finals of top tournaments
    "FIFA World Cup": 60,
    # Continental championships & intercontinental
    "UEFA Euro": 50,
    "Copa América": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Gold Cup": 50,
    "Oceania Nations Cup": 50,
    "Confederations Cup": 50,        # intercontinental
    "CONMEBOL–UEFA Cup of Champions": 50,  # Finalissima
    # Qualifiers for World Cup & continental championships
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 40,
    "AFC Asian Cup qualification": 40,
    "African Cup of Nations qualification": 40,
    "Copa América qualification": 40,
    "Gold Cup qualification": 40,
    "Oceania Nations Cup qualification": 40,
    # Major tournaments (league stages of continental comps, etc.)
    "UEFA Nations League": 40,
    "CONCACAF Nations League": 40,
    "CONCACAF Nations League qualification": 40,
    # All other tournaments
    "Friendly": 20,
}

def get_K(tournament_name: str) -> int:
    """Return Elo K-weight for a given tournament string."""
    # Exact match
    if tournament_name in K_WEIGHTS:
        return K_WEIGHTS[tournament_name]
    # Fallback: guess by keywords
    t = tournament_name.lower()
    if "world cup" in t and "qualif" in t:
        return 40
    if "world cup" in t:
        return 60
    if any(comp in t for comp in ["euro", "copa américa", "asian cup",
                                  "african cup", "gold cup", "oceania"]):
        if "qualif" in t:
            return 40
        return 50
    if "nations league" in t:
        return 40
    if "friendly" in t:
        return 20
    # Default for all other tournaments (King's Cup, etc.)
    return 30

def goal_difference_multiplier(goal_diff: int) -> float:
    """Official World Football Elo goal-difference multiplier."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    # goal_diff >= 3
    return (11 + goal_diff) / 8.0
