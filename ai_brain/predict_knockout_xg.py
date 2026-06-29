#!/usr/bin/env python3
"""
FIFA World Cup 2026 – Knockout Stage xG & Probability Lookup Table Generator
================================================================================
- Symmetric xG via two‑pass averaging.
- Calibrated win/draw/loss probabilities using multinomial logistic regression.
- Mirroring during calibration guarantees perfect symmetry.
- Debug CSV with all features and probabilities.
(Overfitting counter‑measures applied: pruned features, stronger regularization,
 early stopping, feature masking, reduced capacity. FIFA Diff kept.)
Output: knockout_xg_lookup.csv + knockout_prob_lookup.csv + prediction_debug.csv
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os, sys, time, pickle, warnings, random
from collections import defaultdict
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', module='sklearn')
pd.set_option('future.no_silent_downcasting', True)

# =========================================================================
# CONFIGURATION
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../Training Data")) + "/"
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data")) + "/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORCE_FRESH = False                # Set to True to delete all caches and retrain from scratch

MODEL_PATH = f"{OUTPUT_DIR}best_model.pt"
DEBUG_FILE = f"{OUTPUT_DIR}team_features_debug.txt"
TEAM_MAP_FILE = f"{OUTPUT_DIR}team_mapping.pkl"
ELO_CACHE_FILE = f"{OUTPUT_DIR}elo_state_cache.pkl"
SQUAD_CACHE_FILE = f"{OUTPUT_DIR}squad_cache.pkl"
ALIVE_FILE = f"{OUTPUT_DIR}alive_teams.txt"

HISTORY_LEN = 10
EMBEDDING_DIM = 16
MIN_YEAR = 2000
CURRENT_YEAR = 2026
REF_DATE = pd.Timestamp('2026-06-01')
TEMPORAL_DECAY_LAMBDA = 0.15         # steeper decay for recency
EPOCHS = 100
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
N_JOBS = -1
NUM_CORES = mp.cpu_count()

if FORCE_FRESH:
    for f in [MODEL_PATH, DEBUG_FILE, TEAM_MAP_FILE, ELO_CACHE_FILE, SQUAD_CACHE_FILE]:
        if os.path.exists(f):
            os.remove(f)
    print("Forced fresh training – old caches deleted.")

# =========================================================================
# NAME MAP
# =========================================================================
name_map = {
    'USA': 'United States', 'IR Iran': 'Iran', 'Korea Republic': 'South Korea',
    'Congo DR': 'DR Congo', 'Curacao': 'Curaçao', 'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Côte d\'Ivoire': 'Ivory Coast', 'Korea DPR': 'North Korea', 'RCS': 'Czech Republic',
    'Zaire': 'DR Congo', 'Yugoslavia': 'Serbia', 'Netherlands Antilles': 'Curaçao',
    'Türkiye': 'Turkey', 'Korea, South': 'South Korea', 'Cote d\'Ivoire': 'Ivory Coast',
    'Ivory Coast': 'Ivory Coast', 'Iran, Islamic Republic of': 'Iran',
}

# =========================================================================
# TOURNAMENT GROUPS
# =========================================================================
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
team_to_group = {team: grp for grp, teams in GROUPS.items() for team in teams}
ALL_TEAMS = sorted(team_to_group.keys())

# =========================================================================
# 1. LOAD & PREPARE DATA
# =========================================================================
t_start = time.time()
print("Loading data...")
raw_results = pd.read_csv(f"{DATA_DIR}results.csv")
players_df   = pd.read_csv(f"{DATA_DIR}players.csv")
valuations_df = pd.read_csv(f"{DATA_DIR}player_valuations.csv")
fifa_df = pd.read_csv(f"{DATA_DIR}fifa_ranking.csv")

valuations_df['date'] = pd.to_datetime(valuations_df['date'])
raw_results.rename(str.title, axis='columns', inplace=True)
raw_results.rename(columns={'Home_Team': 'Home Team', 'Away_Team': 'Away Team',
                            'Home_Score': 'Home Score', 'Away_Score': 'Away Score'}, inplace=True)
raw_results['Date'] = pd.to_datetime(raw_results['Date'])
raw_results['Home Team'] = raw_results['Home Team'].replace(name_map)
raw_results['Away Team'] = raw_results['Away Team'].replace(name_map)
players_df['country_of_citizenship'] = players_df['country_of_citizenship'].astype(str).str.strip()
players_df['country_of_citizenship'] = players_df['country_of_citizenship'].replace(name_map)

results_df = raw_results[raw_results['Date'].dt.year >= MIN_YEAR].reset_index(drop=True)
results_df = results_df.dropna(subset=['Home Score', 'Away Score'])
results_df['Year'] = results_df['Date'].dt.year
results_df['Weight'] = np.exp(-TEMPORAL_DECAY_LAMBDA * (CURRENT_YEAR - results_df['Year']))
print(f"  Loaded {len(results_df):,} historical matches. ({time.time()-t_start:.1f}s)")

# =========================================================================
# 2. FIFA RANKINGS (used for squad fallback AND as a feature)
# =========================================================================
print("Processing FIFA rankings...")
fifa_df = fifa_df[['rank_date', 'country_full', 'total_points']].copy()
fifa_df['rank_date'] = pd.to_datetime(fifa_df['rank_date'])
fifa_df.rename(columns={'country_full': 'Team', 'total_points': 'fifa_points'}, inplace=True)
fifa_df['Team'] = fifa_df['Team'].replace(name_map)
fifa_df = fifa_df.dropna(subset=['fifa_points'])
fifa_df = fifa_df.sort_values(['Team', 'rank_date']).drop_duplicates(['Team', 'rank_date'], keep='last')
fifa_dict = {}
for team, group in fifa_df.groupby('Team'):
    fifa_dict[team] = group.sort_values('rank_date')

def get_fifa_points(team, match_date):
    if team not in fifa_dict:
        return 1500.0
    prior = fifa_dict[team][fifa_dict[team]['rank_date'] <= match_date]
    if len(prior) == 0:
        return 1500.0
    return prior.iloc[-1]['fifa_points']

# =========================================================================
# 3. MARKET VALUE PREDICTOR
# =========================================================================
print("Training market value predictor...")
value_predictor = None
latest_val_all = (valuations_df[valuations_df['date'] <= REF_DATE]
                  .sort_values('date')
                  .groupby('player_id')
                  .last()
                  .reset_index())
latest_val_clean = latest_val_all[['player_id', 'market_value_in_eur']].copy()
players_for_merge = players_df.copy()
if 'market_value_in_eur' in players_for_merge.columns:
    players_for_merge = players_for_merge.drop(columns=['market_value_in_eur'])
players_with_val = players_for_merge.merge(latest_val_clean, on='player_id', how='left')
has_value = players_with_val['market_value_in_eur'].notna() & (players_with_val['market_value_in_eur'] > 0)
train_data = players_with_val[has_value].copy()

if len(train_data) >= 1000:
    train_data['age'] = (REF_DATE - pd.to_datetime(train_data['date_of_birth'])).dt.days / 365.25
    train_data['position'] = train_data['position'].fillna('Missing')
    le_pos = LabelEncoder()
    train_data['pos_code'] = le_pos.fit_transform(train_data['position'])
    train_data['league_id'] = train_data['current_club_domestic_competition_id'].fillna('Unknown')
    le_league = LabelEncoder()
    train_data['league_code'] = le_league.fit_transform(train_data['league_id'].astype(str))
    train_data['caps'] = train_data['international_caps'].fillna(0).clip(0, 150)
    train_data['height'] = train_data['height_in_cm'].fillna(180).clip(150, 210)
    X_train = train_data[['age', 'pos_code', 'league_code', 'caps', 'height']].fillna(0)
    y_train = np.log1p(train_data['market_value_in_eur'])
    rf_model = RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    value_predictor = {'model': rf_model, 'le_pos': le_pos, 'le_league': le_league}
    print(f"  ✅ Value predictor trained on {len(train_data):,} players")
else:
    print("  ⚠️  Not enough data, predictor disabled")

def batch_predict_market_values(missing_players_df, predictor, ref_date):
    if missing_players_df.empty or predictor is None:
        return np.array([])
    df = missing_players_df.copy()
    df['age'] = (ref_date - pd.to_datetime(df['date_of_birth'])).dt.days / 365.25
    df['position'] = df['position'].fillna('Missing')
    known_pos = set(predictor['le_pos'].classes_)
    df['pos_code'] = df['position'].apply(
        lambda x: predictor['le_pos'].transform(['Missing'])[0] if x not in known_pos
        else predictor['le_pos'].transform([x])[0]
    )
    df['league_id'] = df['current_club_domestic_competition_id'].fillna('Unknown').astype(str)
    known_league = set(predictor['le_league'].classes_)
    df['league_code'] = df['league_id'].apply(
        lambda x: predictor['le_league'].transform(['Unknown'])[0] if x not in known_league
        else predictor['le_league'].transform([x])[0]
    )
    df['caps'] = df['international_caps'].fillna(0).clip(0, 150)
    df['height'] = df['height_in_cm'].fillna(180).clip(150, 210)
    X = df[['age', 'pos_code', 'league_code', 'caps', 'height']].fillna(0)
    return np.expm1(predictor['model'].predict(X))

# =========================================================================
# 4. SQUAD VALUATIONS (with disk caching)
# =========================================================================
print("Preparing squad data...")
valuations_with_country = valuations_df.merge(
    players_df[['player_id', 'country_of_citizenship']], on='player_id', how='left'
).dropna(subset=['country_of_citizenship'])
valuations_with_country['date'] = pd.to_datetime(valuations_with_country['date'])
valuations_with_country['year'] = valuations_with_country['date'].dt.year

def squad_for_country_up_to_year(country, ref_year):
    country_players = players_df[players_df['country_of_citizenship'] == country]
    if country_players.empty:
        fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
        estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
        return {'sum': max(30.0, estimated_sum), 'median': max(1.5, estimated_sum/20),
                'var': 0.0, 'max': max(2.0, estimated_sum/20), 'count_above_50M': 0}
    player_ids = country_players['player_id'].unique()
    val_subset = valuations_df[
        (valuations_df['player_id'].isin(player_ids)) &
        (valuations_df['date'] <= pd.Timestamp(f'{ref_year}-06-01'))
    ]
    latest_vals = (val_subset.sort_values('date').groupby('player_id').last().reset_index()
                   if not val_subset.empty else pd.DataFrame(columns=['player_id','market_value_in_eur']))
    squad_df = country_players[['player_id']].merge(latest_vals, on='player_id', how='left')
    squad_df['market_value_in_eur'] = pd.to_numeric(squad_df['market_value_in_eur'], errors='coerce')
    market_vals = squad_df['market_value_in_eur'].fillna(0.0) / 1_000_000.0
    if (market_vals <= 0).any() and value_predictor is not None:
        missing_mask = market_vals <= 0
        missing_players = country_players[country_players['player_id'].isin(
            squad_df.loc[missing_mask, 'player_id'])]
        if not missing_players.empty:
            predicted = batch_predict_market_values(missing_players, value_predictor,
                                                    pd.Timestamp(f'{ref_year}-06-01'))
            pred_map = dict(zip(missing_players['player_id'], predicted / 1_000_000.0))
            for pid, val in pred_map.items():
                idx = squad_df[squad_df['player_id'] == pid].index
                if len(idx) > 0:
                    market_vals.loc[idx[0]] = val
    known_vals = market_vals[market_vals > 0]
    if len(known_vals) == 0:
        fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
        estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
        return {'sum': max(30.0, estimated_sum), 'median': max(1.5, estimated_sum/20),
                'var': 0.0, 'max': max(2.0, estimated_sum/20), 'count_above_50M': 0}
    if (market_vals <= 0).any():
        market_vals = market_vals.where(market_vals > 0, known_vals.median())
    top23 = market_vals.nlargest(23)
    if len(top23) < 23:
        pad_val = known_vals.median() if len(known_vals) > 0 else 0.5
        top23 = pd.concat([top23, pd.Series([pad_val] * (23 - len(top23)))])
    return {'sum': float(top23.sum()), 'median': float(top23.median()),
            'var': float(top23.var()) if len(top23) > 1 else 0.0,
            'max': float(top23.max()), 'count_above_50M': int((top23 > 50).sum())}

if os.path.exists(SQUAD_CACHE_FILE):
    print("Loading squad cache from disk...")
    with open(SQUAD_CACHE_FILE, 'rb') as f:
        squad_cache = pickle.load(f)
    print(f"  Loaded {len(squad_cache)} cached squad entries.")
else:
    print("Precomputing squad values for all (team, year) pairs in parallel...")
    unique_teams = sorted(set(results_df['Home Team'].unique()) | set(results_df['Away Team'].unique()))
    unique_years = sorted(results_df['Year'].unique())
    pairs = list(product(unique_teams, unique_years))
    total_combos = len(pairs)

    print("  Pre-grouping player data by team...")
    team_player_ids = {}
    for team in unique_teams:
        team_players = players_df[players_df['country_of_citizenship'] == team]
        team_player_ids[team] = team_players['player_id'].unique() if not team_players.empty else np.array([])

    print("  Pre-filtering valuations by player groups...")
    team_valuations = {}
    for team, pids in team_player_ids.items():
        if len(pids) > 0:
            team_valuations[team] = valuations_df[valuations_df['player_id'].isin(pids)].copy()
        else:
            team_valuations[team] = pd.DataFrame(columns=valuations_df.columns)

    def squad_for_country_fast(country, ref_year):
        country_players = players_df[players_df['country_of_citizenship'] == country]
        if country_players.empty:
            fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
            estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
            return {'sum': max(30.0, estimated_sum), 'median': max(1.5, estimated_sum/20),
                    'var': 0.0, 'max': max(2.0, estimated_sum/20), 'count_above_50M': 0}
        val_subset = team_valuations[country][
            team_valuations[country]['date'] <= pd.Timestamp(f'{ref_year}-06-01')
        ]
        latest_vals = (val_subset.sort_values('date').groupby('player_id').last().reset_index()
                       if not val_subset.empty else pd.DataFrame(columns=['player_id','market_value_in_eur']))
        squad_df = country_players[['player_id']].merge(latest_vals, on='player_id', how='left')
        squad_df['market_value_in_eur'] = pd.to_numeric(squad_df['market_value_in_eur'], errors='coerce')
        market_vals = squad_df['market_value_in_eur'].fillna(0.0) / 1_000_000.0
        if (market_vals <= 0).any() and value_predictor is not None:
            missing_mask = market_vals <= 0
            missing_players = country_players[country_players['player_id'].isin(
                squad_df.loc[missing_mask, 'player_id'])]
            if not missing_players.empty:
                predicted = batch_predict_market_values(missing_players, value_predictor,
                                                        pd.Timestamp(f'{ref_year}-06-01'))
                pred_map = dict(zip(missing_players['player_id'], predicted / 1_000_000.0))
                for pid, val in pred_map.items():
                    idx = squad_df[squad_df['player_id'] == pid].index
                    if len(idx) > 0:
                        market_vals.loc[idx[0]] = val
        known_vals = market_vals[market_vals > 0]
        if len(known_vals) == 0:
            fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
            estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
            return {'sum': max(30.0, estimated_sum), 'median': max(1.5, estimated_sum/20),
                    'var': 0.0, 'max': max(2.0, estimated_sum/20), 'count_above_50M': 0}
        if (market_vals <= 0).any():
            market_vals = market_vals.where(market_vals > 0, known_vals.median())
        top23 = market_vals.nlargest(23)
        if len(top23) < 23:
            pad_val = known_vals.median() if len(known_vals) > 0 else 0.5
            top23 = pd.concat([top23, pd.Series([pad_val] * (23 - len(top23)))])
        return {'sum': float(top23.sum()), 'median': float(top23.median()),
                'var': float(top23.var()) if len(top23) > 1 else 0.0,
                'max': float(top23.max()), 'count_above_50M': int((top23 > 50).sum())}

    t0 = time.time()
    results = Parallel(n_jobs=12, backend='loky', verbose=10, batch_size=50)(
        delayed(squad_for_country_fast)(team, year)
        for team, year in pairs
    )
    squad_cache = {p: r for p, r in zip(pairs, results)}
    print(f"\n  Squad cache built in {time.time()-t0:.1f}s")
    current_squad = {team: squad_cache.get((team, CURRENT_YEAR), squad_for_country_up_to_year(team, CURRENT_YEAR)) for team in ALL_TEAMS}
    print("  Current squad data prepared.")

# =========================================================================
# 5. MODEL DEFINITION (static_dim = 7, FIFA Diff included)
# =========================================================================
class PoissonLoss(nn.Module):
    def forward(self, pred, target, weights=None):
        pred = torch.clamp(pred, min=0.1, max=6.0)
        loss = pred - target * torch.log(pred + 1e-8)
        loss = loss.sum(dim=1)
        if weights is not None:
            loss = loss * weights
        return loss.mean()

class OrderInvariantPredictor(nn.Module):
    def __init__(self, num_teams, embed_dim=16, hist_len=10, hist_input_dim=4, static_dim=7):
        super().__init__()
        self.team_embedding = nn.Embedding(num_teams, embed_dim)
        self.emb_dropout = nn.Dropout(0.1)          # new embedding dropout
        self.hist_proj = nn.Linear(hist_input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True, dropout=0.2)
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Smaller static branch
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU()
        )
        fusion_dim = embed_dim * 2 + 32 * 2 + 16
        # Halved MLP sizes + higher dropout
        self.final_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 2), nn.Softplus()
        )

    def forward(self, teamA_id, teamB_id, teamA_seq, teamB_seq, static):
        emb_A = self.emb_dropout(self.team_embedding(teamA_id))
        emb_B = self.emb_dropout(self.team_embedding(teamB_id))
        state_A = self.hist_encoder(self.hist_proj(teamA_seq))[:, -1, :]
        state_B = self.hist_encoder(self.hist_proj(teamB_seq))[:, -1, :]
        static_out = self.static_branch(static)
        combined = torch.cat([emb_A, emb_B, state_A, state_B, static_out], dim=1)
        return self.final_mlp(combined)

# =========================================================================
# 6. FULL TRAINING PIPELINE (with mirroring & anti‑overfitting)
# =========================================================================
def train_and_save_artifacts():
    print("\n[REQUIRED FILES MISSING] Starting full model training...")

    all_teams = sorted(set(results_df['Home Team'].unique()) | set(results_df['Away Team'].unique()))
    team_to_idx_local = {team: i for i, team in enumerate(all_teams)}
    for team in team_to_group:
        if team not in team_to_idx_local:
            all_teams.append(team)
            team_to_idx_local[team] = len(all_teams) - 1
    num_teams = len(all_teams)

    print(f"Computing Elo from scratch for {num_teams} teams...")
    elo_train = defaultdict(lambda: 1500.0)
    team_history_train = defaultdict(list)
    match_data_train = []
    K = 32
    total = len(results_df)
    start_t = time.time()
    for idx in range(total):
        if idx % 5000 == 0:
            print(f"  Elo: {idx}/{total}...")
        row = results_df.iloc[idx]
        h, a = row['Home Team'], row['Away Team']
        year = row['Year']
        match_date = row['Date']
        h_elo_before = elo_train[h]
        a_elo_before = elo_train[a]
        home_fifa = get_fifa_points(h, match_date)
        away_fifa = get_fifa_points(a, match_date)

        def make_seq(team):
            hist = team_history_train[team][-HISTORY_LEN:] if team in team_history_train else []
            seq = []
            for m in hist:
                seq.append([m['goals_for'], m['goals_against'], m['opponent_elo'],
                            1.0 if m['was_home'] else 0.0])
            while len(seq) < HISTORY_LEN:
                seq.insert(0, [0.0, 0.0, 1500.0, 0.0])
            return seq[-HISTORY_LEN:]
        h_seq = make_seq(h)
        a_seq = make_seq(a)

        h_squad = squad_cache[(h, year)]
        a_squad = squad_cache[(a, year)]

        # 7 features: Elo Diff, Squad Sum, Median, Var, Max, Count>50M, FIFA Diff
        feat = [
            h_elo_before - a_elo_before,
            (h_squad['sum'] - a_squad['sum']) / 100.0,
            h_squad['median'] - a_squad['median'],
            np.log1p(h_squad['var'] + 1) - np.log1p(a_squad['var'] + 1),
            (h_squad['max'] - a_squad['max']) / 100.0,
            h_squad['count_above_50M'] - a_squad['count_above_50M'],
            home_fifa - away_fifa,
        ]

        match_info = {
            'teamA_idx': team_to_idx_local[h],
            'teamB_idx': team_to_idx_local[a],
            'teamA_seq': np.array(h_seq, dtype=np.float32),
            'teamB_seq': np.array(a_seq, dtype=np.float32),
            'features': np.array(feat, dtype=np.float32),
            'target_A_goals': row['Home Score'],
            'target_B_goals': row['Away Score'],
            'weight': row['Weight'],
            'year': year,
        }
        match_data_train.append(match_info)

        # Update Elo (neutral ground)
        h_score, a_score = row['Home Score'], row['Away Score']
        if h_score > a_score: h_res, a_res = 1, 0
        elif h_score < a_score: h_res, a_res = 0, 1
        else: h_res, a_res = 0.5, 0.5
        h_exp = 1 / (1 + 10**((a_elo_before - h_elo_before)/400))
        a_exp = 1 / (1 + 10**((h_elo_before - a_elo_before)/400))
        K_adj = K * (1 + min(abs(h_score-a_score), 4)/10)
        elo_train[h] += K_adj * (h_res - h_exp)
        elo_train[a] += K_adj * (a_res - a_exp)
        team_history_train[h].append({'goals_for': h_score, 'goals_against': a_score,
                                      'opponent_elo': a_elo_before, 'was_home': False})
        team_history_train[a].append({'goals_for': a_score, 'goals_against': h_score,
                                      'opponent_elo': h_elo_before, 'was_home': False})
    print(f"  Elo training finished in {time.time()-start_t:.1f}s")

    # =====================================================================
    # BUILD FEATURE MATRICES WITH MIRRORING (7 features)
    # =====================================================================
    features_orig, targets_orig, weights_orig, years_orig = [], [], [], []
    seqA_orig, seqB_orig, idxA_orig, idxB_orig = [], [], [], []
    for m in match_data_train:
        features_orig.append(m['features'])
        targets_orig.append([m['target_A_goals'], m['target_B_goals']])
        weights_orig.append(m['weight'])
        years_orig.append(m['year'])
        seqA_orig.append(m['teamA_seq'])
        seqB_orig.append(m['teamB_seq'])
        idxA_orig.append(m['teamA_idx'])
        idxB_orig.append(m['teamB_idx'])

    # Mirrored copies – all 7 signs flipped
    features_mirror, targets_mirror, weights_mirror, years_mirror = [], [], [], []
    seqA_mirror, seqB_mirror, idxA_mirror, idxB_mirror = [], [], [], []
    for i, m in enumerate(match_data_train):
        feat = -m['features']
        features_mirror.append(feat)
        targets_mirror.append([m['target_B_goals'], m['target_A_goals']])
        weights_mirror.append(m['weight'])
        years_mirror.append(m['year'])
        seqA_mirror.append(m['teamB_seq'])
        seqB_mirror.append(m['teamA_seq'])
        idxA_mirror.append(m['teamB_idx'])
        idxB_mirror.append(m['teamA_idx'])

    features_all = np.array(features_orig + features_mirror, dtype=np.float32)
    targets_all = np.array(targets_orig + targets_mirror, dtype=np.float32)
    weights_all = np.array(weights_orig + weights_mirror, dtype=np.float32)
    years_all = np.array(years_orig + years_mirror)
    seqA_all = np.stack(seqA_orig + seqA_mirror)
    seqB_all = np.stack(seqB_orig + seqB_mirror)
    idxA_all = np.array(idxA_orig + idxA_mirror)
    idxB_all = np.array(idxB_orig + idxB_mirror)

    scaler_train = StandardScaler()
    features_scaled = scaler_train.fit_transform(features_all)

    # Time-based split
    train_mask = years_all < 2022
    val_mask   = (years_all >= 2022) & (years_all < 2025)
    print(f"Training samples: {train_mask.sum()}, Validation samples: {val_mask.sum()}")

    device_train = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_train = OrderInvariantPredictor(num_teams, EMBEDDING_DIM, HISTORY_LEN, 4,
                                          features_scaled.shape[1]).to(device_train)
    criterion = PoissonLoss()
    optimizer = optim.Adam(model_train.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # stronger weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    def create_dataloader(idA, idB, sA, sB, static, y, w, bs, shuffle=True):
        dataset = TensorDataset(
            torch.tensor(idA, dtype=torch.long), torch.tensor(idB, dtype=torch.long),
            torch.tensor(sA, dtype=torch.float32), torch.tensor(sB, dtype=torch.float32),
            torch.tensor(static, dtype=torch.float32), torch.tensor(y, dtype=torch.float32),
            torch.tensor(w, dtype=torch.float32))
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=False, num_workers=0)

    train_loader = create_dataloader(
        idxA_all[train_mask], idxB_all[train_mask], seqA_all[train_mask], seqB_all[train_mask],
        features_scaled[train_mask], targets_all[train_mask], weights_all[train_mask], BATCH_SIZE, True)
    val_loader = create_dataloader(
        idxA_all[val_mask], idxB_all[val_mask], seqA_all[val_mask], seqB_all[val_mask],
        features_scaled[val_mask], targets_all[val_mask], weights_all[val_mask], BATCH_SIZE, False)

    # ---- Feature masking helper ----
    def feature_mask_batch(x, mask_prob=0.15):
        mask = torch.rand(x.shape, device=x.device) > mask_prob
        return x * mask.float()
    # -------------------------------

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    best_model_state = None
    print("Training model...")
    for epoch in range(EPOCHS):
        model_train.train()
        train_loss = 0.0
        for batch in train_loader:
            idA, idB, sA, sB, stat, yb, wb = [x.to(device_train) for x in batch]
            stat = feature_mask_batch(stat, 0.15)   # mask 15% of static features
            optimizer.zero_grad()
            pred = model_train(idA, idB, sA, sB, stat)
            loss = criterion(pred, yb, wb)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        model_train.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                idA, idB, sA, sB, stat, yb, wb = [x.to(device_train) for x in batch]
                # No masking on validation
                val_loss += criterion(model_train(idA, idB, sA, sB, stat), yb, wb).item()
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model_train.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        scheduler.step(val_loss)

    model_train.load_state_dict(best_model_state)
    torch.save(best_model_state, MODEL_PATH)
    print(f"  Training complete. Best val loss: {best_val_loss:.4f}")

    # -----------------------------------------------------------------
    # PERMUTATION FEATURE IMPORTANCE (7 features)
    # -----------------------------------------------------------------
    print("  Computing permutation feature importance on validation set...")
    feature_names = [
        "Elo Diff", "Squad Sum Diff", "Squad Median Diff", "Squad Var Diff",
        "Squad Max Diff", "Count >50M Diff", "FIFA Diff"
    ]

    model_train.eval()
    with torch.no_grad():
        base_loss = 0.0
        for batch in val_loader:
            idA, idB, sA, sB, stat, yb, wb = [x.to(device_train) for x in batch]
            base_loss += criterion(model_train(idA, idB, sA, sB, stat), yb, wb).item()
        base_loss /= len(val_loader)

    importances = {}
    static_val = features_scaled[val_mask].copy()

    for i, fname in enumerate(feature_names):
        static_shuf = static_val.copy()
        np.random.shuffle(static_shuf[:, i])
        temp_dataset = TensorDataset(
            torch.tensor(idxA_all[val_mask], dtype=torch.long),
            torch.tensor(idxB_all[val_mask], dtype=torch.long),
            torch.tensor(seqA_all[val_mask], dtype=torch.float32),
            torch.tensor(seqB_all[val_mask], dtype=torch.float32),
            torch.tensor(static_shuf, dtype=torch.float32),
            torch.tensor(targets_all[val_mask], dtype=torch.float32),
            torch.tensor(weights_all[val_mask], dtype=torch.float32))
        temp_loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)

        perm_loss = 0.0
        with torch.no_grad():
            for batch in temp_loader:
                idA, idB, sA, sB, stat, yb, wb = [x.to(device_train) for x in batch]
                perm_loss += criterion(model_train(idA, idB, sA, sB, stat), yb, wb).item()
        perm_loss /= len(temp_loader)
        importances[fname] = max(0.0, perm_loss - base_loss)
        print(f"    {fname:25s} importance: {importances[fname]:.6f}")

    imp_file = f"{OUTPUT_DIR}feature_importance.txt"
    with open(imp_file, 'w') as f:
        f.write("============================================================\n")
        f.write("FEATURE IMPORTANCE (Permutation Loss Increase)\n")
        f.write("============================================================\n")
        f.write(f"Baseline validation loss: {base_loss:.6f}\n\n")
        f.write(f"{'Feature':30s} Importance\n")
        f.write("-" * 42 + "\n")
        for fname, imp in sorted(importances.items(), key=lambda x: x[1]):
            f.write(f"{fname:30s} {imp:.6f}\n")
        f.write("-" * 42 + "\n")
        f.write("\nSorted by importance (lowest to highest).\n")
    print(f"  Feature importance saved to {imp_file}")

    with open(TEAM_MAP_FILE, 'wb') as f:
        pickle.dump(all_teams, f)

    with open(DEBUG_FILE, 'w') as f:
        f.write("ELO RATINGS (as of 2026-06-01)\n")
        for team in sorted(team_to_group.keys()):
            f.write(f"{team}: {elo_train.get(team, 1500.0):.2f}\n")
        f.write("\nCURRENT SQUAD SUMS & MEDIANS (M€)\n")
        for team in sorted(team_to_group.keys()):
            s = current_squad[team]
            f.write(f"{team}: Sum = {s['sum']:.2f} | Median = {s['median']:.2f}\n")
    print("Artifacts generated successfully.\n")

# =========================================================================
# CHECK & GENERATE MISSING ARTIFACTS
# =========================================================================
if not os.path.exists(MODEL_PATH) or not os.path.exists(DEBUG_FILE) or not os.path.exists(TEAM_MAP_FILE):
    print("\n[WARNING] Model, debug file, or team mapping missing. Running training...")
    train_and_save_artifacts()
else:
    print("All required artifacts found, skipping training.")

# =========================================================================
# 7. LOAD TRAINED MODEL & TEAM MAPPING
# =========================================================================
print("Loading team mapping and model...")
with open(TEAM_MAP_FILE, 'rb') as f:
    team_list = pickle.load(f)
team_to_idx_model = {team: i for i, team in enumerate(team_list)}
num_teams = len(team_list)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OrderInvariantPredictor(num_teams, EMBEDDING_DIM, HISTORY_LEN, 4, 7).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"  Model loaded ({num_teams} teams).")

# =========================================================================
# 8. DYNAMIC ELO & MATCH DATA (7 features)
# =========================================================================
print("Computing Elo ratings (dynamic cache)...")
elo = defaultdict(lambda: 1500.0)
team_history = defaultdict(list)
match_data = []
start_idx = 0
K = 32

if os.path.exists(ELO_CACHE_FILE):
    print("  Loading cached Elo state...")
    with open(ELO_CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
        elo.update(cache['elo'])
        team_history.update(cache['team_history'])
        match_data = cache['match_data']
        start_idx = cache['last_index']
    print(f"  Resumed from match index {start_idx} (Found {len(results_df) - start_idx} new matches).")
else:
    print("  No cache found. Computing historical Elo from scratch...")

total_matches = len(results_df)
if start_idx < total_matches:
    start_elo = time.time()
    new_matches = total_matches - start_idx
    bar_len = 30
    for count, idx in enumerate(range(start_idx, total_matches)):
        if count % 500 == 0 or idx == total_matches - 1:
            elapsed = time.time() - start_elo
            pct = (count + 1) / new_matches * 100
            est_total = elapsed / (count + 1) * new_matches if count > 0 else 0
            remaining = est_total - elapsed
            filled_len = int(bar_len * (count + 1) // new_matches)
            bar = '█' * filled_len + '░' * (bar_len - filled_len)
            sys.stdout.write(f'\r  Elo: [{bar}] {pct:.1f}% | {count+1}/{new_matches} | elapsed: {elapsed:.0f}s | remaining: {remaining:.0f}s ')
            sys.stdout.flush()
        row = results_df.iloc[idx]
        h, a = row['Home Team'], row['Away Team']
        year = row['Year']
        match_date = row['Date']
        h_elo_before = elo[h]
        a_elo_before = elo[a]
        home_fifa = get_fifa_points(h, match_date)
        away_fifa = get_fifa_points(a, match_date)

        def make_seq(team):
            hist = team_history[team][-HISTORY_LEN:] if team in team_history else []
            seq = []
            for m in hist:
                seq.append([m['goals_for'], m['goals_against'], m['opponent_elo'], 1.0 if m['was_home'] else 0.0])
            while len(seq) < HISTORY_LEN:
                seq.insert(0, [0.0, 0.0, 1500.0, 0.0])
            return seq[-HISTORY_LEN:]
        h_seq = make_seq(h)
        a_seq = make_seq(a)

        h_squad = squad_cache[(h, year)]
        a_squad = squad_cache[(a, year)]

        # 7 features
        feat = [
            h_elo_before - a_elo_before,
            (h_squad['sum'] - a_squad['sum']) / 100.0,
            h_squad['median'] - a_squad['median'],
            np.log1p(h_squad['var'] + 1) - np.log1p(a_squad['var'] + 1),
            (h_squad['max'] - a_squad['max']) / 100.0,
            h_squad['count_above_50M'] - a_squad['count_above_50M'],
            home_fifa - away_fifa,
        ]
        match_info = {
            'teamA_seq': np.array(h_seq, dtype=np.float32),
            'teamB_seq': np.array(a_seq, dtype=np.float32),
            'features': np.array(feat, dtype=np.float32),
            'target_A': row['Home Score'],
            'target_B': row['Away Score'],
            'weight': row['Weight'],
            'year': year,
            'teamA_name': h,
            'teamB_name': a,
        }
        match_data.append(match_info)
        h_score, a_score = row['Home Score'], row['Away Score']
        if h_score > a_score: h_res, a_res = 1, 0
        elif h_score < a_score: h_res, a_res = 0, 1
        else: h_res, a_res = 0.5, 0.5
        h_exp = 1 / (1 + 10**((a_elo_before - h_elo_before)/400))
        a_exp = 1 / (1 + 10**((h_elo_before - a_elo_before)/400))
        K_adj = K * (1 + min(abs(h_score-a_score), 4)/10)
        elo[h] += K_adj * (h_res - h_exp)
        elo[a] += K_adj * (a_res - a_exp)
        team_history[h].append({'goals_for': h_score, 'goals_against': a_score,
                                'opponent_elo': a_elo_before, 'was_home': False})
        team_history[a].append({'goals_for': a_score, 'goals_against': h_score,
                                'opponent_elo': h_elo_before, 'was_home': False})
    print(f"\n  Elo computation finished. Total time: {time.time()-start_elo:.1f}s")
    with open(ELO_CACHE_FILE, 'wb') as f:
        pickle.dump({'elo': dict(elo), 'team_history': dict(team_history),
                     'match_data': match_data, 'last_index': total_matches}, f)
else:
    print("  No new matches found. Elo state up to date.")

# =========================================================================
# 9. BUILD SCALER (7 features)
# =========================================================================
print("Fitting scaler...")
features_all = np.array([m['features'] for m in match_data], dtype=np.float32)
scaler = StandardScaler()
scaler.fit(features_all)
print("  Scaler fitted.")

# =========================================================================
# 10. BATCH PREDICTION HELPERS (7 features)
# =========================================================================
def encode_history_batch(teams):
    seqs = []
    for team in teams:
        hist = team_history.get(team, [])
        seq = []
        for m in hist[-HISTORY_LEN:]:
            seq.append([m['goals_for'], m['goals_against'], m['opponent_elo'], 1.0 if m['was_home'] else 0.0])
        while len(seq) < HISTORY_LEN:
            seq.insert(0, [0.0, 0.0, 1500.0, 0.0])
        seqs.append(seq[-HISTORY_LEN:])
    return np.array(seqs, dtype=np.float32)

def build_features_batch(team_a_list, team_b_list):
    feats = []
    for team_a, team_b in zip(team_a_list, team_b_list):
        squad_a = current_squad[team_a]; squad_b = current_squad[team_b]
        fifa_a = get_fifa_points(team_a, REF_DATE); fifa_b = get_fifa_points(team_b, REF_DATE)
        feat = [
            elo[team_a] - elo[team_b],
            (squad_a['sum'] - squad_b['sum']) / 100.0,
            squad_a['median'] - squad_b['median'],
            np.log1p(squad_a['var']+1) - np.log1p(squad_b['var']+1),
            (squad_a['max'] - squad_b['max']) / 100.0,
            squad_a['count_above_50M'] - squad_b['count_above_50M'],
            fifa_a - fifa_b,
        ]
        feats.append(feat)
    return np.array(feats, dtype=np.float32)

# =========================================================================
# 11. PLAYED 2026 MATCHES
# =========================================================================
print("Checking for already played 2026 matches...")
wc_mask = (raw_results['Tournament'].str.contains('World Cup', case=False, na=False)) & \
          (~raw_results['Tournament'].str.contains('qualification', case=False, na=False)) & \
          (raw_results['Date'].dt.year == 2026) & \
          (raw_results['Date'] > pd.Timestamp('2026-06-28'))
wc_2026 = raw_results[wc_mask]
played_pairs = {}
for _, row in wc_2026.iterrows():
    if pd.notna(row['Home Score']) and pd.notna(row['Away Score']):
        played_pairs[(row['Home Team'], row['Away Team'])] = (int(row['Home Score']), int(row['Away Score']))
print(f"  Found {len(played_pairs)} played matches.")

# =========================================================================
# 12. ALIVE TEAMS
# =========================================================================
alive_teams = []
if os.path.exists(ALIVE_FILE):
    with open(ALIVE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2 and parts[1] == '0' and parts[0] in team_to_group:
                alive_teams.append(parts[0])
    if not alive_teams:
        alive_teams = ALL_TEAMS
else:
    alive_teams = ALL_TEAMS
print(f"Alive teams: {len(alive_teams)}")

# =========================================================================
# 13. TRAIN OUTCOME CALIBRATION MODEL (symmetric xG → Win/Draw/Loss)
# =========================================================================
print("\nTraining outcome calibration model on historical data...")
cal_matches = [m for m in match_data if m['year'] >= 2019 and 'teamA_name' in m]
if len(cal_matches) == 0:
    cal_matches = [m for m in match_data if 'teamA_name' in m]

print(f"  Using {len(cal_matches)} matches for calibration.")
teamA_names = [m['teamA_name'] for m in cal_matches]
teamB_names = [m['teamB_name'] for m in cal_matches]

# Predict symmetric xG for calibration matches
fwd_feats = np.array([m['features'] for m in cal_matches], dtype=np.float32)
fwd_feats_scaled = scaler.transform(fwd_feats)
fwd_seqA = np.array([m['teamA_seq'] for m in cal_matches], dtype=np.float32)
fwd_seqB = np.array([m['teamB_seq'] for m in cal_matches], dtype=np.float32)
fwd_idA = [team_to_idx_model[n] for n in teamA_names]
fwd_idB = [team_to_idx_model[n] for n in teamB_names]

# Reversed order (7 features all negated)
rev_feats = -fwd_feats
rev_feats_scaled = scaler.transform(rev_feats)
rev_seqA = fwd_seqB
rev_seqB = fwd_seqA
rev_idA = fwd_idB
rev_idB = fwd_idA

all_feats = np.vstack([fwd_feats_scaled, rev_feats_scaled])
all_seqA = np.vstack([fwd_seqA, rev_seqA])
all_seqB = np.vstack([fwd_seqB, rev_seqB])
all_idA = np.array(fwd_idA + rev_idA)
all_idB = np.array(fwd_idB + rev_idB)

with torch.no_grad():
    preds = model(torch.tensor(all_idA, device=device),
                  torch.tensor(all_idB, device=device),
                  torch.tensor(all_seqA, device=device),
                  torch.tensor(all_seqB, device=device),
                  torch.tensor(all_feats, device=device)).cpu().numpy()

N_cal = len(cal_matches)
pred_AB_cal = preds[:N_cal]
pred_BA_cal = preds[N_cal:]

xG_A_sym = (pred_AB_cal[:, 0] + pred_BA_cal[:, 1]) / 2.0
xG_B_sym = (pred_AB_cal[:, 1] + pred_BA_cal[:, 0]) / 2.0

y_cal_orig = []
for i, m in enumerate(cal_matches):
    if m['target_A'] > m['target_B']:
        y_cal_orig.append(0)
    elif m['target_A'] < m['target_B']:
        y_cal_orig.append(2)
    else:
        y_cal_orig.append(1)
y_cal_orig = np.array(y_cal_orig)

X_cal_mirror = np.column_stack([xG_B_sym, xG_A_sym])
y_cal_mirror = np.where(y_cal_orig == 0, 2, np.where(y_cal_orig == 2, 0, 1))
X_cal_sym = np.vstack([np.column_stack([xG_A_sym, xG_B_sym]), X_cal_mirror])
y_cal_sym = np.concatenate([y_cal_orig, y_cal_mirror])

X_cal_ext = np.column_stack([
    X_cal_sym[:, 0], X_cal_sym[:, 1],
    X_cal_sym[:, 0] - X_cal_sym[:, 1],
    X_cal_sym[:, 0] + X_cal_sym[:, 1],
    (X_cal_sym[:, 0] - X_cal_sym[:, 1]) ** 2,
    X_cal_sym[:, 0] * X_cal_sym[:, 1]
])

cal_model = LogisticRegression(solver='lbfgs', max_iter=1000)
cal_model.fit(X_cal_ext, y_cal_sym)
print(f"  Calibration model trained on {len(X_cal_sym)} symmetrical samples.")

if len(y_cal_sym) > 0:
    from sklearn.metrics import accuracy_score
    pred_labels = cal_model.predict(X_cal_ext)
    acc = accuracy_score(y_cal_sym, pred_labels)
    print(f"  Calibration accuracy on training data: {acc:.3f}")

# =========================================================================
# 14. SYMMETRIC xG & CALIBRATED PROBABILITIES FOR ALL KNOCKOUT PAIRS
# =========================================================================
print("\nGenerating symmetric xG & calibrated probabilities for knockout pairs...")
t_pred = time.time()

unique_pairs = []
for i in range(len(alive_teams)):
    for j in range(i + 1, len(alive_teams)):
        unique_pairs.append((alive_teams[i], alive_teams[j]))

teamA_list = [p[0] for p in unique_pairs]
teamB_list = [p[1] for p in unique_pairs]

all_teamA = teamA_list + teamB_list
all_teamB = teamB_list + teamA_list

features_batch = build_features_batch(all_teamA, all_teamB)
features_scaled = scaler.transform(features_batch)
seq_batch_A = encode_history_batch(all_teamA)
seq_batch_B = encode_history_batch(all_teamB)

feat_tensor = torch.tensor(features_scaled, dtype=torch.float32, device=device)
seqA_tensor = torch.tensor(seq_batch_A, dtype=torch.float32, device=device)
seqB_tensor = torch.tensor(seq_batch_B, dtype=torch.float32, device=device)
idA_tensor = torch.tensor([team_to_idx_model[t] for t in all_teamA], dtype=torch.long, device=device)
idB_tensor = torch.tensor([team_to_idx_model[t] for t in all_teamB], dtype=torch.long, device=device)

with torch.no_grad():
    predictions = model(idA_tensor, idB_tensor, seqA_tensor, seqB_tensor, feat_tensor).cpu().numpy()

N = len(unique_pairs)
pred_AB = predictions[:N]
pred_BA = predictions[N:]

rows_xg = []
rows_prob = []
debug_data = []

for idx, (team_a, team_b) in enumerate(unique_pairs):
    if (team_a, team_b) in played_pairs:
        hg, ag = played_pairs[(team_a, team_b)]
        xG_a, xG_b = hg, ag
        is_played = 1
        if hg > ag:
            p_home, p_draw, p_away = 1.0, 0.0, 0.0
        elif ag > hg:
            p_home, p_draw, p_away = 0.0, 0.0, 1.0
        else:
            p_home, p_draw, p_away = 0.0, 1.0, 0.0
    elif (team_b, team_a) in played_pairs:
        hg, ag = played_pairs[(team_b, team_a)]
        xG_a, xG_b = ag, hg
        is_played = 1
        if ag > hg:
            p_home, p_draw, p_away = 0.0, 0.0, 1.0
        elif hg > ag:
            p_home, p_draw, p_away = 1.0, 0.0, 0.0
        else:
            p_home, p_draw, p_away = 0.0, 1.0, 0.0
    else:
        xG_a = (pred_AB[idx][0] + pred_BA[idx][1]) / 2.0
        xG_b = (pred_AB[idx][1] + pred_BA[idx][0]) / 2.0
        is_played = 0

        X_dir = np.array([[xG_a, xG_b,
                           xG_a - xG_b,
                           xG_a + xG_b,
                           (xG_a - xG_b) ** 2,
                           xG_a * xG_b]])
        probs_dir = cal_model.predict_proba(X_dir)[0]
        X_swap = np.array([[xG_b, xG_a,
                            xG_b - xG_a,
                            xG_b + xG_a,
                            (xG_b - xG_a) ** 2,
                            xG_b * xG_a]])
        probs_swap = cal_model.predict_proba(X_swap)[0]

        p_home = (probs_dir[0] + probs_swap[2]) / 2.0
        p_draw = (probs_dir[1] + probs_swap[1]) / 2.0
        p_away = (probs_dir[2] + probs_swap[0]) / 2.0

    rows_xg.append({
        'Home_Team': team_a,
        'Away_Team': team_b,
        'Home_xG': round(float(xG_a), 2),
        'Away_xG': round(float(xG_b), 2),
        'Is_Played': is_played
    })
    rows_prob.append({
        'Home_Team': team_a,
        'Away_Team': team_b,
        'Home_xG': round(float(xG_a), 2),
        'Away_xG': round(float(xG_b), 2),
        'p_Home_Win': round(float(p_home), 6),
        'p_Draw': round(float(p_draw), 6),
        'p_Away_Win': round(float(p_away), 6),
        'Is_Played': is_played
    })

    squad_a = current_squad[team_a]; squad_b = current_squad[team_b]
    debug_entry = {
        'Team_A': team_a,
        'Team_B': team_b,
        'Elo_A': elo[team_a], 'Elo_B': elo[team_b],
        'Elo_Diff': elo[team_a] - elo[team_b],
        'Squad_Sum_A': squad_a['sum'], 'Squad_Sum_B': squad_b['sum'],
        'Squad_Sum_Diff': (squad_a['sum'] - squad_b['sum']) / 100.0,
        'Squad_Median_Diff': squad_a['median'] - squad_b['median'],
        'Squad_Var_Diff': np.log1p(squad_a['var']+1) - np.log1p(squad_b['var']+1),
        'Squad_Max_Diff': (squad_a['max'] - squad_b['max']) / 100.0,
        'Count_50M_Diff': squad_a['count_above_50M'] - squad_b['count_above_50M'],
        'FIFA_Diff': features_batch[idx][6],
        'xG_A_forward': float(pred_AB[idx][0]),
        'xG_B_forward': float(pred_AB[idx][1]),
        'xG_A_reversed': float(pred_BA[idx][1]),
        'xG_B_reversed': float(pred_BA[idx][0]),
        'xG_A_final': float(xG_a),
        'xG_B_final': float(xG_b),
        'p_Home_Win': p_home,
        'p_Draw': p_draw,
        'p_Away_Win': p_away,
    }
    debug_data.append(debug_entry)

# Save outputs
out_xg_df = pd.DataFrame(rows_xg)
out_xg_csv = f"{OUTPUT_DIR}knockout_xg_lookup.csv"
out_xg_df.to_csv(out_xg_csv, index=False)
print(f"✅ xG lookup table saved ({len(out_xg_df)} rows) -> {out_xg_csv}")

out_prob_df = pd.DataFrame(rows_prob)
out_prob_csv = f"{OUTPUT_DIR}knockout_prob_lookup.csv"
out_prob_df.to_csv(out_prob_csv, index=False)
print(f"✅ Probability lookup table saved ({len(out_prob_df)} rows) -> {out_prob_csv}")

debug_df = pd.DataFrame(debug_data)
debug_csv = f"{OUTPUT_DIR}prediction_debug.csv"
debug_df.to_csv(debug_csv, index=False)
print(f"✅ Detailed debug file saved -> {debug_csv}")

print(f"Total runtime: {time.time()-t_start:.1f}s")