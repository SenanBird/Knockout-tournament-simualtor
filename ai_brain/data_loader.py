# data_loader.py
"""
Data ingestion, cleaning, FIFA ranking extraction, market value imputation,
and squad valuation helpers.

Exposes:
    load_raw_datasets
    process_fifa_rankings
    get_fifa_points
    train_market_value_predictor
    batch_predict_market_values
    build_squad_cache
    load_transfermarkt_values
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

import config

# ---------------------------------------------------------------------------
# Module‑level state (populated by process_fifa_rankings)
# ---------------------------------------------------------------------------
fifa_dict = {}

# ---------------------------------------------------------------------------
# 1. Raw data loading & cleaning
# ---------------------------------------------------------------------------
def load_raw_datasets():
    """Read the four main CSV files and return cleaned DataFrames."""
    raw_results = pd.read_csv(config.DATA_DIR + "results.csv")
    players_df   = pd.read_csv(config.DATA_DIR + "players.csv")
    valuations_df = pd.read_csv(config.DATA_DIR + "player_valuations.csv")
    fifa_df = pd.read_csv(config.DATA_DIR + "fifa_ranking.csv")

    # Timestamps
    valuations_df['date'] = pd.to_datetime(valuations_df['date'])

    # Standardise results columns
    raw_results.rename(str.title, axis='columns', inplace=True)
    raw_results.rename(columns={
        'Home_Team': 'Home Team', 'Away_Team': 'Away Team',
        'Home_Score': 'Home Score', 'Away_Score': 'Away Score'
    }, inplace=True)
    raw_results['Date'] = pd.to_datetime(raw_results['Date'])

    # Apply name mapping
    raw_results['Home Team'] = raw_results['Home Team'].replace(config.name_map)
    raw_results['Away Team'] = raw_results['Away Team'].replace(config.name_map)
    players_df['country_of_citizenship'] = (
        players_df['country_of_citizenship']
        .astype(str).str.strip().replace(config.name_map)
    )

    # Filter to MIN_YEAR and drop rows without scores
    results_df = raw_results[
        raw_results['Date'].dt.year >= config.MIN_YEAR
    ].reset_index(drop=True).dropna(subset=['Home Score', 'Away Score'])

    # Add year and temporal weight
    results_df['Year'] = results_df['Date'].dt.year
    results_df['Weight'] = np.exp(
        -config.TEMPORAL_DECAY_LAMBDA * (config.CURRENT_YEAR - results_df['Year'])
    )

    return results_df, players_df, valuations_df, fifa_df


# ---------------------------------------------------------------------------
# 2. FIFA rankings
# ---------------------------------------------------------------------------
def process_fifa_rankings(fifa_df):
    """Populate the global fifa_dict for fast point lookups."""
    global fifa_dict
    fifa_df = fifa_df[['rank_date', 'country_full', 'total_points']].copy()
    fifa_df['rank_date'] = pd.to_datetime(fifa_df['rank_date'])
    fifa_df.rename(columns={'country_full': 'Team', 'total_points': 'fifa_points'}, inplace=True)
    fifa_df['Team'] = fifa_df['Team'].replace(config.name_map)
    fifa_df = fifa_df.dropna(subset=['fifa_points']).sort_values(['Team', 'rank_date'])
    fifa_dict.clear()
    for team, group in fifa_df.groupby('Team'):
        fifa_dict[team] = group.sort_values('rank_date')


def get_fifa_points(team, match_date):
    """Return FIFA points for a team as of a given date (default 1500)."""
    if team not in fifa_dict:
        return 1500.0
    prior = fifa_dict[team][fifa_dict[team]['rank_date'] <= match_date]
    if len(prior) == 0:
        return 1500.0
    return float(prior.iloc[-1]['fifa_points'])


# ---------------------------------------------------------------------------
# 3. Market value imputation (Random Forest)
# ---------------------------------------------------------------------------
def train_market_value_predictor(players_df, valuations_df):
    """
    Train a Random Forest to predict log(market_value) from basic player attributes.
    Returns a dict with model, label encoders, or None if insufficient data.
    """
    # Latest valuation per player before REF_DATE
    latest_val_all = (
        valuations_df[valuations_df['date'] <= config.REF_DATE]
        .sort_values('date')
        .groupby('player_id')
        .last()
        .reset_index()
    )
    latest_val_clean = latest_val_all[['player_id', 'market_value_in_eur']].copy()
    players_for_merge = players_df.copy()
    if 'market_value_in_eur' in players_for_merge.columns:
        players_for_merge = players_for_merge.drop(columns=['market_value_in_eur'])
    players_with_val = players_for_merge.merge(latest_val_clean, on='player_id', how='left')

    has_value = players_with_val['market_value_in_eur'].notna() & (players_with_val['market_value_in_eur'] > 0)
    train_data = players_with_val[has_value].copy()

    if len(train_data) < 1000:
        print("  ⚠️  Not enough data for value predictor (<1000 samples).")
        return None

    train_data['age'] = (config.REF_DATE - pd.to_datetime(train_data['date_of_birth'])).dt.days / 365.25
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

    predictor = {'model': rf_model, 'le_pos': le_pos, 'le_league': le_league}
    print(f"  ✅ Value predictor trained on {len(train_data):,} players")
    return predictor


def batch_predict_market_values(missing_players_df, predictor, ref_date):
    """
    Predict market values for a DataFrame of players that have missing values.
    Returns array of predicted values in EUR.
    """
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


# ---------------------------------------------------------------------------
# 4. Squad valuation for a single country & year (fast, for parallel use)
# ---------------------------------------------------------------------------
def _squad_for_country_fast(country, ref_year, players_df, team_valuations, predictor):
    """
    Compute {sum, median, var, max, count_above_50M, age_avg, age_var, caps_sum}
    for the best 23 players of a country in a given year.
    Uses pre‑filtered team_valuations.
    """
    country_players = players_df[players_df['country_of_citizenship'] == country]
    if country_players.empty:
        fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
        estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
        return {
            'sum': max(30.0, estimated_sum),
            'median': max(1.5, estimated_sum / 20),
            'var': 0.0,
            'max': max(2.0, estimated_sum / 20),
            'count_above_50M': 0,
            'age_avg': 26.0,
            'age_var': 4.0,
            'caps_sum': 0.0,
        }

    val_subset = team_valuations[country][
        team_valuations[country]['date'] <= pd.Timestamp(f'{ref_year}-06-01')
    ]
    latest_vals = (
        val_subset.sort_values('date').groupby('player_id').last().reset_index()
        if not val_subset.empty
        else pd.DataFrame(columns=['player_id', 'market_value_in_eur'])
    )

    # Build squad DataFrame and RESET INDEX
    squad_df = country_players[['player_id']].merge(latest_vals, on='player_id', how='left')
    squad_df = squad_df.reset_index(drop=True)

    squad_df['market_value_in_eur'] = pd.to_numeric(squad_df['market_value_in_eur'], errors='coerce')
    market_vals = squad_df['market_value_in_eur'].fillna(0.0) / 1_000_000.0

    # Impute missing values with the predictor
    if (market_vals <= 0).any() and predictor is not None:
        missing_mask = market_vals <= 0
        missing_players = country_players[country_players['player_id'].isin(
            squad_df.loc[missing_mask, 'player_id']
        )]
        if not missing_players.empty:
            predicted = batch_predict_market_values(
                missing_players, predictor,
                pd.Timestamp(f'{ref_year}-06-01')
            )
            pred_map = dict(zip(missing_players['player_id'], predicted / 1_000_000.0))
            for pid, val in pred_map.items():
                idx = squad_df[squad_df['player_id'] == pid].index
                if len(idx) > 0:
                    market_vals.loc[idx[0]] = val

    known_vals = market_vals[market_vals > 0]
    if len(known_vals) == 0:
        fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
        estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
        return {
            'sum': max(30.0, estimated_sum),
            'median': max(1.5, estimated_sum / 20),
            'var': 0.0,
            'max': max(2.0, estimated_sum / 20),
            'count_above_50M': 0,
            'age_avg': 26.0,
            'age_var': 4.0,
            'caps_sum': 0.0,
        }

    if (market_vals <= 0).any():
        market_vals = market_vals.where(market_vals > 0, known_vals.median())

    # Select top 23 real players
    top23_raw = market_vals.nlargest(23)
    actual_indices = top23_raw.index   # indices of real players in squad_df

    # Pad to exactly 23 values if necessary (synthetic players for statistics only)
    if len(top23_raw) < 23:
        pad_val = known_vals.median() if len(known_vals) > 0 else 0.5
        start_pad = actual_indices.max() + 1 if len(actual_indices) > 0 else 0
        pad_indices = range(start_pad, start_pad + (23 - len(top23_raw)))
        pad_series = pd.Series([pad_val] * (23 - len(top23_raw)), index=pad_indices)
        top23 = pd.concat([top23_raw, pad_series])
    else:
        top23 = top23_raw

    # Compute age stats and caps ONLY on real players
    top23_player_ids = squad_df.loc[actual_indices, 'player_id']
    top23_players = country_players[country_players['player_id'].isin(top23_player_ids)].copy()

    if not top23_players.empty:
        ref_date = pd.Timestamp(f'{ref_year}-06-01')
        top23_players['age'] = (
            ref_date - pd.to_datetime(top23_players['date_of_birth'])
        ).dt.days / 365.25
        age_avg = float(top23_players['age'].mean())
        age_var = float(top23_players['age'].var()) if len(top23_players) > 1 else 0.0
        caps_sum = float(top23_players['international_caps'].fillna(0).sum())
    else:
        age_avg = 26.0
        age_var = 4.0
        caps_sum = 0.0

    return {
        'sum': float(top23.sum()),
        'median': float(top23.median()),
        'var': float(top23.var()) if len(top23) > 1 else 0.0,
        'max': float(top23.max()),
        'count_above_50M': int((top23 > 50).sum()),
        'age_avg': age_avg,
        'age_var': age_var,
        'caps_sum': caps_sum,
    }


# ---------------------------------------------------------------------------
# 5. Build / load the squad cache (disk caching & parallel execution)
# ---------------------------------------------------------------------------
def build_squad_cache(players_df, valuations_df, teams, years, predictor, n_jobs=-1):
    """
    Compute squad values for all (team, year) combinations and return a dict.
    Uses joblib for parallelism and writes the result to SQUAD_CACHE_FILE.
    """
    # Pre‑filter valuations by team for speed
    team_valuations = {}
    for team in teams:
        country_players = players_df[players_df['country_of_citizenship'] == team]
        if not country_players.empty:
            pids = country_players['player_id'].unique()
            team_valuations[team] = valuations_df[valuations_df['player_id'].isin(pids)].copy()
        else:
            team_valuations[team] = pd.DataFrame(columns=valuations_df.columns)

    pairs = [(team, year) for team in teams for year in years]

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10, batch_size=50)(
        delayed(_squad_for_country_fast)(team, year, players_df, team_valuations, predictor)
        for team, year in pairs
    )

    squad_cache = {p: r for p, r in zip(pairs, results)}

    # Save to disk for future reuse
    with open(config.SQUAD_CACHE_FILE, 'wb') as f:
        pickle.dump(squad_cache, f)
    print(f"  Squad cache built and saved to {config.SQUAD_CACHE_FILE}")

    return squad_cache


def load_squad_cache_from_disk():
    """Load previously saved squad cache, if it exists."""
    if os.path.exists(config.SQUAD_CACHE_FILE):
        with open(config.SQUAD_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# 6. Transfermarkt actual values override
# ---------------------------------------------------------------------------
def load_transfermarkt_values(current_squad):
    """
    Overwrite estimated squad values with actual Transfermarkt totals/averages
    for the 2026 tournament teams.  Returns the updated dict.
    """
    if not os.path.exists(config.TRANSFERMARKT_FILE):
        print(f"  ⚠️  {config.TRANSFERMARKT_FILE} not found. Using estimated squad values.")
        return current_squad

    tm_df = pd.read_csv(config.TRANSFERMARKT_FILE)
    tm_df['Team'] = tm_df['Team'].replace(config.name_map)

    # Convert EUR to millions
    tm_df['TotalValueM'] = tm_df['TotalValueEUR'] / 1_000_000.0
    tm_df['AvgValueM'] = tm_df['AvgValueEUR'] / 1_000_000.0

    updated = 0
    for _, row in tm_df.iterrows():
        team = row['Team']
        if team in current_squad:
            current_squad[team]['sum'] = row['TotalValueM']
            current_squad[team]['median'] = row['AvgValueM']  # using average as proxy for median
            updated += 1
        else:
            print(f"  ⚠️  Team '{team}' not in tournament list – skipping")

    print(f"  Overwritten {updated} teams with Transfermarkt values.")
    return current_squad