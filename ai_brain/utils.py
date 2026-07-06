# utils.py
"""
Mathematical helpers, feature engineering, and sequence builders for the
World Cup knockout prediction pipeline.

All functions are self‑contained and receive required state explicitly
(no hidden globals).
"""

import numpy as np
import torch
import config
from data_loader import get_fifa_points


# ---------------------------------------------------------------------------
# 1. Weighted margin of recent performances
# ---------------------------------------------------------------------------
def compute_weighted_margin(team, team_history, decay=0.8):
    """
    Compute a weighted moving average of (goal difference * opponent Elo / 1500)
    over the most recent `config.HISTORY_LEN` matches for `team`.
    Returns 0.0 if no history.
    """
    hist = team_history.get(team, [])
    if not hist:
        return 0.0

    recent = hist[-config.HISTORY_LEN:]
    weights = np.exp(-decay * np.arange(len(recent))[::-1])
    weights /= weights.sum()

    margins = []
    for m in recent:
        margin = (m['goals_for'] - m['goals_against']) * (m['opponent_elo'] / 1500.0)
        margins.append(margin)

    return float(np.average(margins, weights=weights))


# ---------------------------------------------------------------------------
# 2. Sequence builders for the transformer history encoder
# ---------------------------------------------------------------------------
def make_sequence_for_team(team, team_history):
    """
    Build a fixed‑length sequence (config.HISTORY_LEN) of
    [goals_for, goals_against, opponent_elo, was_home] for a single team.
    Pads with [0,0,1500,0] if history is too short.
    """
    hist = team_history.get(team, [])
    seq = []
    for m in hist[-config.HISTORY_LEN:]:
        seq.append([
            m['goals_for'],
            m['goals_against'],
            m['opponent_elo'],
            1.0 if m['was_home'] else 0.0
        ])

    # Pad at the beginning to ensure constant length
    while len(seq) < config.HISTORY_LEN:
        seq.insert(0, [0.0, 0.0, 1500.0, 0.0])

    return seq[-config.HISTORY_LEN:]


def encode_history_batch(teams, team_history):
    """
    Vectorised version returning a numpy array of shape
    (len(teams), config.HISTORY_LEN, 4).
    """
    seqs = [make_sequence_for_team(t, team_history) for t in teams]
    return np.array(seqs, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Static feature vector builder (7 dimensions)
# ---------------------------------------------------------------------------
def build_features_batch(team_a_list, team_b_list, current_squad, elo, team_history,
                         ref_dates=None):
    if ref_dates is None:
        ref_dates = [config.REF_DATE] * len(team_a_list)
    """
    For each pair (team_a, team_b) construct the 7‑dimensional feature vector:
      [elo_diff, sum_diff/100, median_diff, var_diff, count50M_diff,
       fifa_diff, weighted_margin_diff]
    Uses the passed `current_squad` dict (from the squad cache), `elo` dict,
    and `team_history`.

    Returns a float32 array of shape (len(team_a_list), 7).
    """
    feats = []
    for team_a, team_b, ref_date in zip(team_a_list, team_b_list, ref_dates):
        squad_a = current_squad[team_a]
        squad_b = current_squad[team_b]
        fifa_a = get_fifa_points(team_a, ref_date)
        fifa_b = get_fifa_points(team_b, ref_date)
        margin_a = compute_weighted_margin(team_a, team_history)
        margin_b = compute_weighted_margin(team_b, team_history)

        # ----- new features -----
        # age
        age_avg_diff = squad_a['age_avg'] - squad_b['age_avg']
        age_var_diff = squad_a['age_var'] - squad_b['age_var']

        # value per cap
        cap_a = squad_a['caps_sum'] + 1e-6
        cap_b = squad_b['caps_sum'] + 1e-6
        ratio_a = squad_a['sum'] / cap_a
        ratio_b = squad_b['sum'] / cap_b
        ratio_diff = ratio_a - ratio_b

        # days since last match
        days_a = get_days_since_last_match(team_a, team_history, ref_date)
        days_b = get_days_since_last_match(team_b, team_history, ref_date)
        days_diff = days_a - days_b

        feat = [
            elo[team_a] - elo[team_b],
            (squad_a['sum'] - squad_b['sum']) / 100.0,
            squad_a['median'] - squad_b['median'],
            np.log1p(squad_a['var'] + 1) - np.log1p(squad_b['var'] + 1),
            squad_a['count_above_50M'] - squad_b['count_above_50M'],
            fifa_a - fifa_b,
            margin_a - margin_b,
            age_avg_diff,
            age_var_diff,
            days_diff,
            ratio_diff,
        ]
        feats.append(feat)

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. Asymmetric feature dropout (used during training)
# ---------------------------------------------------------------------------
def feature_mask_batch_asymmetric(x, mask_probs):
    """
    Randomly drop out columns of a batch tensor `x` with probabilities
    `mask_probs` (list of length = x.shape[1]).  Returns the masked tensor.
    """
    mask = torch.ones_like(x)
    for i, p in enumerate(mask_probs):
        keep = torch.rand(x.shape[0], device=x.device) > p
        mask[:, i] = keep.float()
    return x * mask


def get_days_since_last_match(team, team_history, ref_date):
    """Return days since team's last match before ref_date, capped at 30."""
    matches = team_history.get(team, [])
    if not matches:
        return 30.0
    # search backwards for the most recent match before ref_date
    for m in reversed(matches):
        if m['date'] < ref_date:
            days = (ref_date - m['date']).days
            return min(days, 30.0)
    return 30.0