# main.py
"""
Orchestration script for the FIFA World Cup 2026 knockout stage
xG & probability lookup table generator.

Usage:
    python main.py

Dependencies: config.py, models.py, data_loader.py, utils.py
"""

import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import psutil
import platform
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


# Local modules
import config
import models
import data_loader
import utils

# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
elo = defaultdict(lambda: 1500.0)
team_history = defaultdict(list)
match_data = []
scaler = None
model = None
team_to_idx_model = {}
team_list = []
current_squad = {}
alive_teams = []


# ---------------------------------------------------------------------------
# SYSTEM / HARDWARE INFORMATION LOGGING
# ---------------------------------------------------------------------------
def print_system_info():
    """Print detailed system and hardware information."""
    print("\n" + "=" * 70)
    print("SYSTEM & HARDWARE INFORMATION")
    print("=" * 70)

    print(f"  Platform:      {platform.platform()}")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  PyTorch:       {torch.__version__}")

    print(f"  CPU cores:     {psutil.cpu_count(logical=True)} logical, "
          f"{psutil.cpu_count(logical=False)} physical")
    ram = psutil.virtual_memory()
    print(f"  RAM total:     {ram.total / (1024**3):.1f} GB")
    print(f"  RAM available: {ram.available / (1024**3):.1f} GB")

    if torch.cuda.is_available():
        print(f"  CUDA:          YES (v{torch.version.cuda})")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / (1024**3)
            print(f"  GPU {i} ({props.name}):")
            print(f"    Total VRAM:      {mem_total:.1f} GB")
            print(f"    Compute:         {props.major}.{props.minor}")
            print(f"    Multi‑processor: {props.multi_processor_count}")
    else:
        print("  CUDA:          NO (using CPU)")

    print("=" * 70 + "\n")


def print_gpu_memory(label=""):
    """Print current GPU memory usage (if CUDA is available)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        pct = (allocated / total) * 100
        print(f"  GPU Memory [{label}]: "
              f"allocated={allocated:.3f} GB, "
              f"cached={cached:.3f} GB, "
              f"total={total:.1f} GB ({pct:.1f}% used)")


# ---------------------------------------------------------------------------
# 1. TRAINING & ARTIFACT GENERATION
# ---------------------------------------------------------------------------
def train_and_save_artifacts(results_df, players_df, valuations_df, squad_cache):
    """Run the full training pipeline."""
    print("\n" + "=" * 70)
    print("STARTING MODEL TRAINING")
    print("=" * 70)

    # Build team index
    all_teams = sorted(set(results_df['Home Team'].unique()) | set(results_df['Away Team'].unique()))
    for team in config.ALL_TEAMS:
        if team not in all_teams:
            all_teams.append(team)
    team_to_idx = {team: i for i, team in enumerate(all_teams)}
    num_teams = len(all_teams)

    print(f"\n  Teams in index:     {num_teams}")
    print(f"  Historical matches: {len(results_df):,}")
    print_gpu_memory("before training")

    # Helper to compute days since last match for a given team before a specific date
    def _days_since_last_match(team, history, ref_date):
        matches = history.get(team, [])
        if not matches:
            return 30.0
        for m in reversed(matches):
            if m['date'] < ref_date:
                days = (ref_date - m['date']).days
                return min(days, 30.0)
        return 30.0

    # Elo computation
    print(f"\nComputing Elo from scratch for {num_teams} teams...")
    elo_train = defaultdict(lambda: 1500.0)
    team_history_train = defaultdict(list)
    match_data_train = []
    K = config.ELO_K
    total = len(results_df)
    start_t = time.time()

    for idx in range(total):
        if idx % 5000 == 0:
            elapsed = time.time() - start_t
            pct = (idx / total) * 100
            print(f"  Elo: {idx:,}/{total:,} ({pct:.1f}%) | {elapsed:.0f}s elapsed")
        row = results_df.iloc[idx]
        h, a = row['Home Team'], row['Away Team']
        year = row['Year']
        match_date = row['Date']
        h_elo_before = elo_train[h]
        a_elo_before = elo_train[a]
        home_fifa = data_loader.get_fifa_points(h, match_date)
        away_fifa = data_loader.get_fifa_points(a, match_date)

        h_seq = utils.make_sequence_for_team(h, team_history_train)
        a_seq = utils.make_sequence_for_team(a, team_history_train)

        h_squad = squad_cache[(h, year)]
        a_squad = squad_cache[(a, year)]
        h_margin = utils.compute_weighted_margin(h, team_history_train)
        a_margin = utils.compute_weighted_margin(a, team_history_train)

        # --- original features ---
        feat = [
            h_elo_before - a_elo_before,
            (h_squad['sum'] - a_squad['sum']) / 100.0,
            h_squad['median'] - a_squad['median'],
            np.log1p(h_squad['var'] + 1) - np.log1p(a_squad['var'] + 1),
            h_squad['count_above_50M'] - a_squad['count_above_50M'],
            home_fifa - away_fifa,
            h_margin - a_margin,
        ]

        # --- new features ---
        # Age
        age_avg_diff = h_squad.get('age_avg', 26.0) - a_squad.get('age_avg', 26.0)
        age_var_diff = h_squad.get('age_var', 4.0) - a_squad.get('age_var', 4.0)

        # Value per cap ratio
        cap_h = h_squad.get('caps_sum', 0.0) + 1e-6
        cap_a = a_squad.get('caps_sum', 0.0) + 1e-6
        ratio_h = h_squad['sum'] / cap_h
        ratio_a = a_squad['sum'] / cap_a
        ratio_diff = ratio_h - ratio_a

        # Days since last match
        days_h = _days_since_last_match(h, team_history_train, match_date)
        days_a = _days_since_last_match(a, team_history_train, match_date)
        days_diff = days_h - days_a

        feat.extend([age_avg_diff, age_var_diff, days_diff, ratio_diff])

        match_info = {
            'teamA_idx': team_to_idx[h],
            'teamB_idx': team_to_idx[a],
            'teamA_seq': np.array(h_seq, dtype=np.float32),
            'teamB_seq': np.array(a_seq, dtype=np.float32),
            'features': np.array(feat, dtype=np.float32),
            'target_A_goals': row['Home Score'],
            'target_B_goals': row['Away Score'],
            'weight': row['Weight'],
            'year': year,
        }
        match_data_train.append(match_info)

        # Elo update
        h_score, a_score = row['Home Score'], row['Away Score']
        if h_score > a_score:
            h_res, a_res = 1, 0
        elif h_score < a_score:
            h_res, a_res = 0, 1
        else:
            h_res, a_res = 0.5, 0.5
        h_exp = 1 / (1 + 10**((a_elo_before - h_elo_before) / 400))
        a_exp = 1 / (1 + 10**((h_elo_before - a_elo_before) / 400))
        K_adj = K * (1 + min(abs(h_score - a_score), 4) / 10)
        elo_train[h] += K_adj * (h_res - h_exp)
        elo_train[a] += K_adj * (a_res - a_exp)

        team_history_train[h].append({
            'goals_for': h_score, 'goals_against': a_score,
            'opponent_elo': a_elo_before, 'was_home': False,
            'date': match_date
        })
        team_history_train[a].append({
            'goals_for': a_score, 'goals_against': h_score,
            'opponent_elo': h_elo_before, 'was_home': False,
            'date': match_date
        })

    elo_time = time.time() - start_t
    print(f"  ✅ Elo training finished in {elo_time:.1f}s")
    print(f"     ({len(match_data_train) / elo_time:.0f} matches/sec)")

    # Build mirrored dataset
    print("\nBuilding mirrored dataset (symmetric xG)...")
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

    features_mirror, targets_mirror, weights_mirror, years_mirror = [], [], [], []
    seqA_mirror, seqB_mirror, idxA_mirror, idxB_mirror = [], [], [], []
    for i, m in enumerate(match_data_train):
        # Mirror features: sign flip for most, but days diff should flip sign, ratio diff also flips, age diffs flip, etc.
        # Since the whole feature vector is a diff (teamA - teamB), we can simply negate it.
        features_mirror.append(-m['features'])
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

    print(f"  Total samples (mirrored): {len(features_all):,}")
    print(f"  Feature dim:              {features_all.shape[1]}")
    print(f"  Sequence shape:           {seqA_all.shape}")

    scaler_train = StandardScaler()
    features_scaled = scaler_train.fit_transform(features_all)

    train_mask = years_all < 2022
    val_mask   = (years_all >= 2022) & (years_all < 2025)
    print(f"\n  Training samples:   {train_mask.sum():,}")
    print(f"  Validation samples: {val_mask.sum():,}")

    # Model instantiation
    print("\nInstantiating model...")
    model_train = models.OrderInvariantPredictor(
        num_teams, config.EMBEDDING_DIM, config.HISTORY_LEN,
        4, static_dim=config.NUM_STATIC_FEATURES
    ).to(device)

    total_params = sum(p.numel() for p in model_train.parameters())
    trainable_params = sum(p.numel() for p in model_train.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print_gpu_memory("after model init")

    criterion = models.PoissonLoss()
    optimizer = optim.Adam(model_train.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
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
        features_scaled[train_mask], targets_all[train_mask], weights_all[train_mask], config.BATCH_SIZE, True)
    val_loader = create_dataloader(
        idxA_all[val_mask], idxB_all[val_mask], seqA_all[val_mask], seqB_all[val_mask],
        features_scaled[val_mask], targets_all[val_mask], weights_all[val_mask], config.BATCH_SIZE, False)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    mask_probs = config.MASK_PROBS

    # Training loop
    print(f"\n{'='*70}")
    print(f"TRAINING (asymmetric dropout: Elo={mask_probs[0]}, others={mask_probs[1]})")
    print(f"{'='*70}")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    best_model_state = None
    train_start = time.time()

    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        model_train.train()
        train_loss = 0.0
        for batch in train_loader:
            idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
            stat = utils.feature_mask_batch_asymmetric(stat, mask_probs)
            optimizer.zero_grad()
            pred = model_train(idA, idB, sA, sB, stat)
            loss = criterion(pred, yb, wb)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model_train.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
                val_loss += criterion(model_train(idA, idB, sA, sB, stat), yb, wb).item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        epoch_time = time.time() - epoch_start

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{config.EPOCHS} | "
                  f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                  f"time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model_train.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break
        scheduler.step(val_loss)

    train_total_time = time.time() - train_start
    print(f"\n  ✅ Training complete in {train_total_time:.1f}s")
    print(f"     Best validation loss: {best_val_loss:.4f}")
    print_gpu_memory("after training")

    # Load the best weights back into the training model object
    model_train.load_state_dict(best_model_state)
    
    # Save the state dict explicitly from the model object
    torch.save(model_train.state_dict(), config.MODEL_PATH)

    model_size_mb = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
    print(f"  Model saved: {config.MODEL_PATH} ({model_size_mb:.1f} MB)")

    # Feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Permutation)")
    print("=" * 70)
    model_train.eval()
    with torch.no_grad():
        base_loss = 0.0
        for batch in val_loader:
            idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
            base_loss += criterion(model_train(idA, idB, sA, sB, stat), yb, wb).item()
        base_loss /= len(val_loader)

    importances = {}
    static_val = features_scaled[val_mask].copy()
    for i, fname in enumerate(config.FEATURE_NAMES):
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
        temp_loader = DataLoader(temp_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
        perm_loss = 0.0
        with torch.no_grad():
            for batch in temp_loader:
                idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
                perm_loss += criterion(model_train(idA, idB, sA, sB, stat), yb, wb).item()
        perm_loss /= len(temp_loader)
        importances[fname] = max(0.0, perm_loss - base_loss)
        print(f"    {fname:25s} importance: {importances[fname]:.6f}")

    imp_file = config.OUTPUT_DIR + "feature_importance.txt"
    with open(imp_file, 'w') as f:
        f.write("============================================================\n")
        f.write("FEATURE IMPORTANCE (Permutation Loss Increase)\n")
        f.write("============================================================\n")
        f.write(f"Baseline validation loss: {base_loss:.6f}\n\n")
        for fname, imp in sorted(importances.items(), key=lambda x: x[1]):
            f.write(f"{fname:30s} {imp:.6f}\n")
    print(f"  Feature importance saved to {imp_file}")

    with open(config.TEAM_MAP_FILE, 'wb') as f:
        pickle.dump(all_teams, f)

    with open(config.DEBUG_FILE, 'w') as f:
        f.write("ELO RATINGS (as of 2026-06-01)\n")
        for team in sorted(config.ALL_TEAMS):
            f.write(f"{team}: {elo_train.get(team, 1500.0):.2f}\n")
        f.write("\nCURRENT SQUAD SUMS & MEDIANS (M€)\n")
        for team in sorted(config.ALL_TEAMS):
            s = squad_cache.get((team, config.CURRENT_YEAR), {'sum':0,'median':0,'age_avg':0,'age_var':0,'caps_sum':0})
            f.write(f"{team}: Sum = {s['sum']:.2f} | Median = {s['median']:.2f} | "
                    f"Age Avg = {s.get('age_avg', 'N/A')} | Caps Sum = {s.get('caps_sum', 'N/A')}\n")

    print("  Artifacts generated successfully.\n")


# ---------------------------------------------------------------------------
# 2. DYNAMIC ELO / MATCH DATA COMPUTATION
# ---------------------------------------------------------------------------
def compute_elo_and_match_data(results_df, squad_cache, start_idx=0):
    """Append new matches to global match_data and update elo & team_history."""
    global elo, team_history, match_data
    total = len(results_df)

    print("\n" + "=" * 70)
    print("DYNAMIC ELO COMPUTATION")
    print("=" * 70)

    # Helper to compute days since last match
    def _days_since_last_match(team, history, ref_date):
        matches = history.get(team, [])
        if not matches:
            return 30.0
        for m in reversed(matches):
            if m['date'] < ref_date:
                days = (ref_date - m['date']).days
                return min(days, 30.0)
        return 30.0

    if os.path.exists(config.ELO_CACHE_FILE):
        print("  Loading cached Elo state...")
        with open(config.ELO_CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
            elo.update(cache['elo'])
            team_history.update(cache['team_history'])
            match_data = cache['match_data']
            start_idx = cache['last_index']
        print(f"  Resumed from match index {start_idx} (Found {total - start_idx} new matches).")
    else:
        print("  No cache found. Computing historical Elo from scratch...")
        elo.clear()
        team_history.clear()
        match_data = []
        start_idx = 0

    if start_idx >= total:
        print("  No new matches found. Elo state up to date.")
        return start_idx

    K = config.ELO_K
    start_elo = time.time()
    new_matches = total - start_idx

    for count, idx in enumerate(range(start_idx, total)):
        if count % 5000 == 0 or idx == total - 1:
            elapsed = time.time() - start_elo
            pct = (count + 1) / new_matches * 100
            rate = (count + 1) / elapsed if elapsed > 0 else 0
            remaining = (new_matches - count - 1) / rate if rate > 0 else 0
            print(f"  Elo: {count+1:,}/{new_matches:,} ({pct:.1f}%) | "
                  f"rate: {rate:.0f} matches/s | remaining: {remaining:.0f}s")

        row = results_df.iloc[idx]
        h, a = row['Home Team'], row['Away Team']
        year = row['Year']
        match_date = row['Date']
        h_elo_before = elo[h]
        a_elo_before = elo[a]
        home_fifa = data_loader.get_fifa_points(h, match_date)
        away_fifa = data_loader.get_fifa_points(a, match_date)

        h_seq = utils.make_sequence_for_team(h, team_history)
        a_seq = utils.make_sequence_for_team(a, team_history)

        h_squad = squad_cache[(h, year)]
        a_squad = squad_cache[(a, year)]
        h_margin = utils.compute_weighted_margin(h, team_history)
        a_margin = utils.compute_weighted_margin(a, team_history)

        # Original features
        feat = [
            h_elo_before - a_elo_before,
            (h_squad['sum'] - a_squad['sum']) / 100.0,
            h_squad['median'] - a_squad['median'],
            np.log1p(h_squad['var'] + 1) - np.log1p(a_squad['var'] + 1),
            h_squad['count_above_50M'] - a_squad['count_above_50M'],
            home_fifa - away_fifa,
            h_margin - a_margin,
        ]

        # New features (with fallback defaults if old squad cache lacks keys)
        age_avg_diff = h_squad.get('age_avg', 26.0) - a_squad.get('age_avg', 26.0)
        age_var_diff = h_squad.get('age_var', 4.0) - a_squad.get('age_var', 4.0)

        cap_h = h_squad.get('caps_sum', 0.0) + 1e-6
        cap_a = a_squad.get('caps_sum', 0.0) + 1e-6
        ratio_h = h_squad['sum'] / cap_h
        ratio_a = a_squad['sum'] / cap_a
        ratio_diff = ratio_h - ratio_a

        days_h = _days_since_last_match(h, team_history, match_date)
        days_a = _days_since_last_match(a, team_history, match_date)
        days_diff = days_h - days_a

        feat.extend([age_avg_diff, age_var_diff, days_diff, ratio_diff])

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

        # Elo update
        h_score, a_score = row['Home Score'], row['Away Score']
        if h_score > a_score:
            h_res, a_res = 1, 0
        elif h_score < a_score:
            h_res, a_res = 0, 1
        else:
            h_res, a_res = 0.5, 0.5
        h_exp = 1 / (1 + 10**((a_elo_before - h_elo_before) / 400))
        a_exp = 1 / (1 + 10**((h_elo_before - a_elo_before) / 400))
        K_adj = K * (1 + min(abs(h_score - a_score), 4) / 10)
        elo[h] += K_adj * (h_res - h_exp)
        elo[a] += K_adj * (a_res - a_exp)

        team_history[h].append({
            'goals_for': h_score, 'goals_against': a_score,
            'opponent_elo': a_elo_before, 'was_home': False,
            'date': match_date
        })
        team_history[a].append({
            'goals_for': a_score, 'goals_against': h_score,
            'opponent_elo': h_elo_before, 'was_home': False,
            'date': match_date
        })

    elo_time = time.time() - start_elo
    print(f"  ✅ Elo computation finished in {elo_time:.1f}s")
    print(f"     ({new_matches / elo_time:.0f} matches/sec)")

    with open(config.ELO_CACHE_FILE, 'wb') as f:
        pickle.dump({
            'elo': dict(elo),
            'team_history': dict(team_history),
            'match_data': match_data,
            'last_index': total
        }, f)
    print(f"  Elo state cached to {config.ELO_CACHE_FILE}")

    return total


# ---------------------------------------------------------------------------
# 3. PROBABILITY LOOKUP GENERATION (Poisson Monte Carlo)
# ---------------------------------------------------------------------------
def generate_probability_lookups(alive_teams, current_squad, results_df):
    """Generate the knockout_prob_lookup.csv and prediction_debug.csv files."""
    global scaler, model, team_to_idx_model, team_list, elo, team_history

    print("\n" + "=" * 70)
    print("PROBABILITY LOOKUP GENERATION")
    print("=" * 70)
    t_pred = time.time()

    # Fit scaler on ALL match features (already computed)
    features_all = np.array([m['features'] for m in match_data], dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(features_all)
    print(f"  Scaler fitted on {len(features_all):,} features")

    # Unique pairs
    unique_pairs = []
    for i in range(len(alive_teams)):
        for j in range(i + 1, len(alive_teams)):
            unique_pairs.append((alive_teams[i], alive_teams[j]))

    n_pairs = len(unique_pairs)
    print(f"  Unique knockout pairs: {n_pairs:,}")
    print(f"  Poisson simulations/pair: {config.POISSON_SIMULATIONS:,}")
    print(f"  Total simulations: {n_pairs * config.POISSON_SIMULATIONS:,}")

    teamA_list = [p[0] for p in unique_pairs]
    teamB_list = [p[1] for p in unique_pairs]

    all_teamA = teamA_list + teamB_list
    all_teamB = teamB_list + teamA_list

    # Provide a list of ref_dates (same REF_DATE for all current teams)
    ref_dates = [config.REF_DATE] * len(all_teamA)
    features_batch = utils.build_features_batch(all_teamA, all_teamB,
                                                current_squad, elo, team_history,
                                                ref_dates=ref_dates)
    features_scaled = scaler.transform(features_batch)
    seq_batch_A = utils.encode_history_batch(all_teamA, team_history)
    seq_batch_B = utils.encode_history_batch(all_teamB, team_history)

    feat_tensor = torch.tensor(features_scaled, dtype=torch.float32, device=device)
    seqA_tensor = torch.tensor(seq_batch_A, dtype=torch.float32, device=device)
    seqB_tensor = torch.tensor(seq_batch_B, dtype=torch.float32, device=device)
    idA_tensor = torch.tensor([team_to_idx_model[t] for t in all_teamA],
                              dtype=torch.long, device=device)
    idB_tensor = torch.tensor([team_to_idx_model[t] for t in all_teamB],
                              dtype=torch.long, device=device)

    print(f"  Inference tensor sizes: {feat_tensor.shape}, {seqA_tensor.shape}")
    print_gpu_memory("before inference")

    with torch.no_grad():
        predictions = model(idA_tensor, idB_tensor, seqA_tensor, seqB_tensor,
                            feat_tensor).cpu().numpy()

    print_gpu_memory("after inference")

    N = len(unique_pairs)
    pred_AB = predictions[:N]
    pred_BA = predictions[N:]

    print(f"  xG range: [{pred_AB.min():.2f}, {pred_AB.max():.2f}]")
    
    # Played matches (Use the already cleaned results_df passed into the function)
    wc_mask = (results_df['Tournament'].str.contains('World Cup', case=False, na=False)) & \
              (~results_df['Tournament'].str.contains('qualification', case=False, na=False)) & \
              (results_df['Date'].dt.year == 2026) & \
              (results_df['Date'] > pd.Timestamp('2026-06-28'))
              
    played_pairs = {}
    for _, row in results_df[wc_mask].iterrows():
        if pd.notna(row['Home Score']) and pd.notna(row['Away Score']):
            played_pairs[(row['Home Team'], row['Away Team'])] = (int(row['Home Score']), int(row['Away Score']))

    rng_prob = np.random.RandomState(config.POISSON_RANDOM_SEED)
    rows_prob = []
    debug_rows = []

    for idx, (team_a, team_b) in enumerate(unique_pairs):
        if (team_a, team_b) in played_pairs:
            hg, ag = played_pairs[(team_a, team_b)]
            xG_a, xG_b = hg, ag
            is_played = 1
            if hg > ag:      p_home, p_draw, p_away = 1.0, 0.0, 0.0
            elif ag > hg:    p_home, p_draw, p_away = 0.0, 0.0, 1.0
            else:            p_home, p_draw, p_away = 0.0, 1.0, 0.0
        elif (team_b, team_a) in played_pairs:
            hg, ag = played_pairs[(team_b, team_a)]
            xG_a, xG_b = ag, hg
            is_played = 1
            if ag > hg:      p_home, p_draw, p_away = 0.0, 0.0, 1.0
            elif hg > ag:    p_home, p_draw, p_away = 1.0, 0.0, 0.0
            else:            p_home, p_draw, p_away = 0.0, 1.0, 0.0
        else:
            xG_a = (pred_AB[idx][0] + pred_BA[idx][1]) / 2.0
            xG_b = (pred_AB[idx][1] + pred_BA[idx][0]) / 2.0
            is_played = 0

            goalsA = rng_prob.poisson(xG_a, config.POISSON_SIMULATIONS)
            goalsB = rng_prob.poisson(xG_b, config.POISSON_SIMULATIONS)
            home_win = np.sum(goalsA > goalsB)
            draw = np.sum(goalsA == goalsB)
            away_win = np.sum(goalsA < goalsB)
            total_sim = config.POISSON_SIMULATIONS
            p_home = home_win / total_sim
            p_draw = draw / total_sim
            p_away = away_win / total_sim

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

        # Debug row now includes all features (indices 0..10)
        f = features_batch[idx]
        debug_rows.append({
            'Team_A': team_a,
            'Team_B': team_b,
            'Elo_Diff': f[0],
            'Squad_Sum_Diff': f[1],
            'Squad_Median_Diff': f[2],
            'Squad_Var_Diff': f[3],
            'Count_50M_Diff': f[4],
            'FIFA_Diff': f[5],
            'Margin_Diff': f[6],
            'Age_Avg_Diff': f[7],
            'Age_Var_Diff': f[8],
            'DaysSinceLast_Diff': f[9],
            'ValuePerCap_Diff': f[10],
            'xG_A_final': float(xG_a),
            'xG_B_final': float(xG_b),
            'p_Home_Win': p_home,
            'p_Draw': p_draw,
            'p_Away_Win': p_away,
        })

    # Write CSVs to OUTPUT_DIR
    out_prob_df = pd.DataFrame(rows_prob)
    out_prob_csv = config.OUTPUT_DIR + "knockout_prob_lookup.csv"
    out_prob_df.to_csv(out_prob_csv, index=False)
    out_prob_size_mb = os.path.getsize(out_prob_csv) / (1024 * 1024)
    print(f"\n  ✅ Probability lookup saved: {out_prob_csv}")
    print(f"     Rows: {len(out_prob_df):,} | Size: {out_prob_size_mb:.2f} MB")

    debug_df = pd.DataFrame(debug_rows)
    debug_csv = config.OUTPUT_DIR + "prediction_debug.csv"
    debug_df.to_csv(debug_csv, index=False)
    debug_size_mb = os.path.getsize(debug_csv) / (1024 * 1024)
    print(f"  ✅ Debug file saved: {debug_csv} ({debug_size_mb:.2f} MB)")

    print(f"\n  Lookup generation finished in {time.time() - t_pred:.1f}s")


# ---------------------------------------------------------------------------
# 4. MAIN ORCHESTRATION
# ---------------------------------------------------------------------------
def main():
    global current_squad, alive_teams, device, model, team_list, team_to_idx_model
    t_start = time.time()

    # --- System info ---
    print_system_info()

    # --- Force fresh if requested ---
    if config.FORCE_FRESH:
        for f in [config.MODEL_PATH, config.DEBUG_FILE, config.TEAM_MAP_FILE,
                  config.ELO_CACHE_FILE, config.SQUAD_CACHE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        print("Forced fresh training – old caches deleted.")

    # --- Load data ---
    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)
    t_load = time.time()
    results_df, players_df, valuations_df, fifa_df = data_loader.load_raw_datasets()
    data_loader.process_fifa_rankings(fifa_df)
    print(f"  Loaded {len(results_df):,} historical matches in {time.time() - t_load:.1f}s")
    print(f"  Players database: {len(players_df):,} records")
    print(f"  Valuations: {len(valuations_df):,} records")
    print(f"  FIFA rankings: {len(fifa_df):,} records")

    # --- Market value predictor ---
    print("\n" + "=" * 70)
    print("MARKET VALUE PREDICTOR")
    print("=" * 70)
    value_predictor = data_loader.train_market_value_predictor(players_df, valuations_df)

    # --- Squad cache ---
    print("\n" + "=" * 70)
    print("SQUAD VALUATION CACHE")
    print("=" * 70)
    squad_cache = data_loader.load_squad_cache_from_disk()
    if squad_cache is None:
        unique_teams = sorted(set(results_df['Home Team'].unique()) | set(results_df['Away Team'].unique()))
        unique_years = sorted(results_df['Year'].unique())
        print(f"  Building cache for {len(unique_teams)} teams × {len(unique_years)} years...")
        squad_cache = data_loader.build_squad_cache(players_df, valuations_df,
                                                    unique_teams, unique_years,
                                                    value_predictor, n_jobs=config.N_JOBS)
    else:
        print(f"  Loaded {len(squad_cache)} cached squad entries.")

    # current_squad with all necessary keys, including new ones
    default_squad = {'sum':0, 'median':0, 'var':0, 'max':0, 'count_above_50M':0,
                     'age_avg': 26.0, 'age_var': 4.0, 'caps_sum': 0.0}
    current_squad = {team: squad_cache.get((team, config.CURRENT_YEAR), default_squad)
                     for team in config.ALL_TEAMS}
    current_squad = data_loader.load_transfermarkt_values(current_squad)

    # --- Train if needed ---
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.TEAM_MAP_FILE):
        train_and_save_artifacts(results_df, players_df, valuations_df, squad_cache)

    # --- Load model artifacts ---
    print("\n" + "=" * 70)
    print("LOADING MODEL ARTIFACTS")
    print("=" * 70)
    with open(config.TEAM_MAP_FILE, 'rb') as f:
        team_list = pickle.load(f)
    team_to_idx_model = {team: i for i, team in enumerate(team_list)}
    num_teams = len(team_list)

    model = models.OrderInvariantPredictor(num_teams, config.EMBEDDING_DIM,
                                           config.HISTORY_LEN, 4,
                                           static_dim=config.NUM_STATIC_FEATURES).to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()
    print(f"  Model loaded ({num_teams} teams).")
    print_gpu_memory("after model load")

    # --- Dynamic Elo & match data ---
    compute_elo_and_match_data(results_df, squad_cache)

    # --- Alive teams ---
    alive_teams = []
    if os.path.exists(config.ALIVE_FILE):
        with open(config.ALIVE_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and parts[1] == '0' and parts[0] in config.team_to_group:
                    alive_teams.append(parts[0])
    if not alive_teams:
        alive_teams = config.ALL_TEAMS
    print(f"\nAlive teams: {len(alive_teams)}")

    # --- Poisson calibration evaluation ---
    print("\n" + "=" * 70)
    print("POISSON CALIBRATION ON HISTORICAL DATA")
    print("=" * 70)
    cal_matches = [m for m in match_data if m['year'] >= 2019 and 'teamA_name' in m]
    if not cal_matches:
        cal_matches = [m for m in match_data if 'teamA_name' in m]

    if cal_matches:
        # Fit scaler on all match features (needed before calibration)
        features_all = np.array([m['features'] for m in match_data], dtype=np.float32)
        scaler = StandardScaler()
        scaler.fit(features_all)
        print(f"  Scaler fitted on {len(features_all):,} features")

        teamA_names = [m['teamA_name'] for m in cal_matches]
        teamB_names = [m['teamB_name'] for m in cal_matches]
        fwd_feats = np.array([m['features'] for m in cal_matches], dtype=np.float32)
        fwd_feats_scaled = scaler.transform(fwd_feats)
        fwd_seqA = np.array([m['teamA_seq'] for m in cal_matches], dtype=np.float32)
        fwd_seqB = np.array([m['teamB_seq'] for m in cal_matches], dtype=np.float32)
        fwd_idA = [team_to_idx_model[n] for n in teamA_names]
        fwd_idB = [team_to_idx_model[n] for n in teamB_names]

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
        xG_A_sym = (preds[:N_cal, 0] + preds[N_cal:, 1]) / 2.0
        xG_B_sym = (preds[:N_cal, 1] + preds[N_cal:, 0]) / 2.0

        # --- xG SYSTEMIC CALIBRATION CHECK ---
        actual_A = np.array([m['target_A'] for m in cal_matches])
        actual_B = np.array([m['target_B'] for m in cal_matches])
        
        total_predicted_xg = np.sum(xG_A_sym) + np.sum(xG_B_sym)
        total_actual_goals = np.sum(actual_A) + np.sum(actual_B)
        
        bias_ratio = total_predicted_xg / total_actual_goals if total_actual_goals > 0 else 0
        
        mae_A = np.abs(xG_A_sym - actual_A)
        mae_B = np.abs(xG_B_sym - actual_B)
        overall_mae = np.mean(np.concatenate([mae_A, mae_B]))

        print(f"  Total Actual Goals:    {total_actual_goals:.0f}")
        print(f"  Total Predicted xG:    {total_predicted_xg:.0f}")
        print(f"  Systemic Bias Ratio:   {bias_ratio:.3f} (1.00 is perfect)")
        print(f"  xG Mean Absolute Err:  {overall_mae:.3f} goals/team")
        print("-" * 40)

        y_cal_orig = []
        for m in cal_matches:
            if m['target_A'] > m['target_B']:        y_cal_orig.append(0)
            elif m['target_A'] < m['target_B']:      y_cal_orig.append(2)
            else:                                     y_cal_orig.append(1)

        rng_cal = np.random.RandomState(config.POISSON_RANDOM_SEED)
        correct = 0
        total_cal = len(cal_matches)
        for i in range(total_cal):
            actual = y_cal_orig[i]
            goalsA = rng_cal.poisson(xG_A_sym[i], config.POISSON_SIMULATIONS)
            goalsB = rng_cal.poisson(xG_B_sym[i], config.POISSON_SIMULATIONS)
            home_win = np.sum(goalsA > goalsB)
            draw = np.sum(goalsA == goalsB)
            away_win = np.sum(goalsA < goalsB)
            probs = np.array([home_win, draw, away_win]) / config.POISSON_SIMULATIONS
            pred_class = np.argmax(probs)
            if pred_class == actual:
                correct += 1

        acc_poisson = correct / total_cal if total_cal > 0 else 0.0
        print(f"  Matches evaluated: {total_cal:,}")
        print(f"  Poisson calibration accuracy: {acc_poisson:.3f} ({correct}/{total_cal})")
    else:
        print("  No calibration matches found, skipping.")

    # --- Final probability lookups ---
    generate_probability_lookups(alive_teams, current_squad, results_df)

    print("\n" + "=" * 70)
    print(f"✅ PIPELINE COMPLETE")
    print(f"   Output CSVs → {config.OUTPUT_DIR}")
    print(f"   Model cache → {config.CACHE_DIR}")
    print(f"   Total runtime: {time.time() - t_start:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()