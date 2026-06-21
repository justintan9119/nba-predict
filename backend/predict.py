import datetime
import math
import re
import time
import warnings
from collections import defaultdict

import pandas as pd
import requests
from pandas.errors import PerformanceWarning
from nba_api.stats.endpoints import leaguegamelog, scoreboardv3
from nba_api.stats.library.parameters import LeagueID, LeagueIDNullable
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from injuries import injury_adjusted_home_probability


LEAGUES = {
    'nba': {
        'league_id': LeagueID.nba,
        'league_id_nullable': LeagueIDNullable.nba,
        'live_boxscore_base_url': 'https://cdn.nba.com/static/json/liveData/boxscore',
        'referer': 'https://www.nba.com/',
    },
    'wnba': {
        'league_id': LeagueID.wnba,
        'league_id_nullable': LeagueIDNullable.wnba,
        'live_boxscore_base_url': 'https://cdn.wnba.com/static/json/liveData/boxscore',
        'referer': 'https://www.wnba.com/',
    },
}

RATE_FEATURES = ['OFF_RATING', 'TS_PCT', 'AST_100', 'REB_100', 'TOV_100', 'STL_100', 'BLK_100']
OPPONENT_FEATURES = ['DEF_RATING', 'OPP_TS_PCT', 'OPP_AST_100', 'OPP_REB_100', 'OPP_TOV_100', 'OPP_STL_100', 'OPP_BLK_100', 'NET_RATING']
RESULT_FEATURES = ['WIN', 'POINT_DIFF']
MODEL_SIGNAL_FEATURES = RATE_FEATURES + OPPONENT_FEATURES + RESULT_FEATURES
ROLLING_WINDOWS = [5, 10, 20]
LOCATION_SPLIT_WINDOW = 10
CONTEXT_FEATURES = ['REST_DAYS', 'B2B', 'THREE_IN_FOUR', 'GAMES_PLAYED', 'ELO_PRE']
ELO_HOME_ADVANTAGE = 65
ELO_K = 20
MODEL_PARAMS = {
    'n_estimators': 180,
    'learning_rate': 0.04,
    'max_depth': 3,
    'subsample': 0.85,
    'random_state': 42,
}

_MODEL_CACHE = {}
_FEATURES_CACHE = {}
_METRICS_CACHE = {}
_SNAPSHOTS_CACHE = {}

warnings.simplefilter('ignore', PerformanceWarning)


def normalize_league(league):
    return league if league in LEAGUES else 'nba'


def get_seasons(num_seasons, league='nba'):
    league = normalize_league(league)
    now = datetime.datetime.now()
    current_year = now.year

    if league == 'nba' and now.month < 10:
        current_year -= 1

    if league == 'wnba':
        if now.month < 5:
            current_year -= 1
        return [str(year) for year in range(current_year - num_seasons + 1, current_year + 1)]

    return [f"{year}-{(year + 1) % 100:02d}" for year in range(current_year - num_seasons + 1, current_year + 1)]


def get_nba_seasons(num_seasons):
    return get_seasons(num_seasons, 'nba')


def snapshot_feature_names():
    names = []
    for window in ROLLING_WINDOWS:
        names.extend([f'R{window}_{feature}' for feature in MODEL_SIGNAL_FEATURES])
    names.extend([f'STD_{feature}' for feature in MODEL_SIGNAL_FEATURES])
    names.extend([f'LOC_R{LOCATION_SPLIT_WINDOW}_{feature}' for feature in MODEL_SIGNAL_FEATURES])
    names.extend([f'LOC_STD_{feature}' for feature in MODEL_SIGNAL_FEATURES])
    names.extend(CONTEXT_FEATURES)
    return names


def add_rate_features(games):
    games = games.copy()
    possessions = games['FGA'] + 0.44 * games['FTA'] - games['OREB'] + games['TOV']
    shot_attempts = games['FGA'] + 0.44 * games['FTA']

    games['POSS'] = possessions.replace(0, pd.NA)
    games['OFF_RATING'] = (games['PTS'] / games['POSS']) * 100
    games['TS_PCT'] = games['PTS'] / (2 * shot_attempts.replace(0, pd.NA))
    games['AST_100'] = (games['AST'] / games['POSS']) * 100
    games['REB_100'] = (games['REB'] / games['POSS']) * 100
    games['TOV_100'] = (games['TOV'] / games['POSS']) * 100
    games['STL_100'] = (games['STL'] / games['POSS']) * 100
    games['BLK_100'] = (games['BLK'] / games['POSS']) * 100
    return games.fillna(0)



def add_opponent_adjusted_features(games):
    opponent = games[['GAME_ID', 'TEAM_ID'] + RATE_FEATURES].copy()
    opponent = opponent.rename(columns={'TEAM_ID': 'OPP_TEAM_ID', **{feature: f'OPP_{feature}' for feature in RATE_FEATURES}})
    games = games.merge(opponent, on='GAME_ID')
    games = games[games['TEAM_ID'] != games['OPP_TEAM_ID']].copy()
    games['DEF_RATING'] = games['OPP_OFF_RATING']
    games['OPP_TS_PCT'] = games['OPP_TS_PCT']
    games['OPP_AST_100'] = games['OPP_AST_100']
    games['OPP_REB_100'] = games['OPP_REB_100']
    games['OPP_TOV_100'] = games['OPP_TOV_100']
    games['OPP_STL_100'] = games['OPP_STL_100']
    games['OPP_BLK_100'] = games['OPP_BLK_100']
    games['NET_RATING'] = games['OFF_RATING'] - games['DEF_RATING']
    return games
def parse_clock(clock):
    clock_match = re.search(r'PT(\d+)M(\d+(?:\.\d+)?)S?', clock or '')
    mins = int(clock_match.group(1)) if clock_match else 0
    secs = int(float(clock_match.group(2))) if clock_match else 0
    return mins, secs, clock_match


def format_clock(clock):
    mins, secs, clock_match = parse_clock(clock)
    return f"{mins}:{secs:02d}" if clock_match else clock


def full_team_name(team):
    city = team.get('teamCity') or team.get('teamCityName') or ''
    name = team.get('teamName') or ''
    return f"{city} {name}".strip() or name

def add_elo_pregame(games):
    games = games.copy().sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID'])
    games['ELO_PRE'] = 1500.0
    elos = defaultdict(lambda: 1500.0)

    for _, game_rows in games.groupby('GAME_ID', sort=False):
        for idx, row in game_rows.iterrows():
            games.at[idx, 'ELO_PRE'] = elos[int(row['TEAM_ID'])]

        if len(game_rows) != 2:
            continue

        home_rows = game_rows[game_rows['IS_HOME'] == 1]
        if home_rows.empty:
            continue

        home = home_rows.iloc[0]
        away = game_rows[game_rows['TEAM_ID'] != home['TEAM_ID']].iloc[0]
        home_id = int(home['TEAM_ID'])
        away_id = int(away['TEAM_ID'])
        home_elo = elos[home_id]
        away_elo = elos[away_id]
        expected_home = 1 / (1 + 10 ** ((away_elo - (home_elo + ELO_HOME_ADVANTAGE)) / 400))
        result_home = int(home['WIN'])
        margin = max(1, abs(float(home.get('POINT_DIFF', 0))))
        margin_multiplier = math.log(margin + 1)
        update = ELO_K * margin_multiplier * (result_home - expected_home)
        elos[home_id] += update
        elos[away_id] -= update

    return games, dict(elos)


def add_rest_context(games):
    games = games.copy().sort_values(['TEAM_ID', 'GAME_DATE', 'GAME_ID'])
    games['REST_DAYS'] = 7.0
    games['B2B'] = 0
    games['THREE_IN_FOUR'] = 0

    for _, group in games.groupby('TEAM_ID', sort=False):
        previous_dates = []
        for idx, row in group.iterrows():
            game_date = row['GAME_DATE']
            if previous_dates:
                rest_days = (game_date - previous_dates[-1]).days
                games.at[idx, 'REST_DAYS'] = max(0, min(rest_days, 7))
                games.at[idx, 'B2B'] = 1 if rest_days <= 1 else 0
                recent_games = sum(1 for date in previous_dates if 0 < (game_date - date).days <= 3)
                games.at[idx, 'THREE_IN_FOUR'] = 1 if recent_games >= 2 else 0
            previous_dates.append(game_date)

    return games


def prepare_pregame_team_rows(games):
    games = games.copy()
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games['IS_HOME'] = games['MATCHUP'].astype(str).str.contains('vs.').astype(int)
    games['WIN'] = games['WL'].map({'W': 1, 'L': 0}).fillna(0).astype(int)
    games['POINT_DIFF'] = games['PLUS_MINUS'] if 'PLUS_MINUS' in games.columns else 0
    games = add_opponent_adjusted_features(add_rate_features(games))
    games = add_rest_context(games)
    games, _ = add_elo_pregame(games)
    games = games.sort_values(['TEAM_ID', 'GAME_DATE', 'GAME_ID'])

    team_group = games.groupby('TEAM_ID', group_keys=False)
    season_keys = ['TEAM_ID', 'SEASON'] if 'SEASON' in games.columns else ['TEAM_ID']
    season_group = games.groupby(season_keys, group_keys=False)
    source_features = MODEL_SIGNAL_FEATURES

    for feature in source_features:
        for window in ROLLING_WINDOWS:
            games[f'R{window}_{feature}'] = team_group[feature].transform(
                lambda values: values.shift().rolling(window, min_periods=1).mean()
            )
        games[f'STD_{feature}'] = season_group[feature].transform(
            lambda values: values.shift().expanding(min_periods=1).mean()
        )

    location_group = games.groupby(['TEAM_ID', 'IS_HOME'], group_keys=False)
    location_season_keys = ['TEAM_ID', 'SEASON', 'IS_HOME'] if 'SEASON' in games.columns else ['TEAM_ID', 'IS_HOME']
    location_season_group = games.groupby(location_season_keys, group_keys=False)
    for feature in source_features:
        games[f'LOC_R{LOCATION_SPLIT_WINDOW}_{feature}'] = location_group[feature].transform(
            lambda values: values.shift().rolling(LOCATION_SPLIT_WINDOW, min_periods=1).mean()
        )
        games[f'LOC_STD_{feature}'] = location_season_group[feature].transform(
            lambda values: values.shift().expanding(min_periods=1).mean()
        )

    games['GAMES_PLAYED'] = season_group.cumcount()
    return games.fillna(0)


def matchup_feature_frame(home_rows, away_rows):
    home = home_rows.reset_index(drop=True)
    away = away_rows.reset_index(drop=True)
    data = {}

    for feature in snapshot_feature_names():
        data[f'DIFF_{feature}'] = home[feature] - away[feature]

    for feature in CONTEXT_FEATURES:
        data[f'HOME_{feature}'] = home[feature]
        data[f'AWAY_{feature}'] = away[feature]

    return pd.DataFrame(data).fillna(0)


def build_training_data(games):
    team_rows = prepare_pregame_team_rows(games)
    home_rows = team_rows[team_rows['IS_HOME'] == 1].copy()
    away_rows = team_rows[team_rows['IS_HOME'] == 0].copy()
    merged = home_rows.merge(
        away_rows[['GAME_ID', 'TEAM_ID'] + snapshot_feature_names()],
        on='GAME_ID',
        suffixes=('_HOME', '_AWAY')
    )

    home_features = merged[[f'{feature}_HOME' for feature in snapshot_feature_names()]].copy()
    away_features = merged[[f'{feature}_AWAY' for feature in snapshot_feature_names()]].copy()
    home_features.columns = snapshot_feature_names()
    away_features.columns = snapshot_feature_names()

    X = matchup_feature_frame(home_features, away_features)
    y = merged['WIN'].astype(int)
    dates = merged['GAME_DATE']
    return X, y, dates


def build_team_snapshots(games, league):
    games = games.copy()
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games['IS_HOME'] = games['MATCHUP'].astype(str).str.contains('vs.').astype(int)
    games['WIN'] = games['WL'].map({'W': 1, 'L': 0}).fillna(0).astype(int)
    games['POINT_DIFF'] = games['PLUS_MINUS'] if 'PLUS_MINUS' in games.columns else 0
    games = add_opponent_adjusted_features(add_rate_features(games)).sort_values(['TEAM_ID', 'GAME_DATE', 'GAME_ID'])
    games, final_elos = add_elo_pregame(games)
    today = pd.Timestamp(datetime.datetime.now().date())
    snapshots = {}

    for team_id, group in games.groupby('TEAM_ID'):
        group = group.sort_values('GAME_DATE')
        snapshot = {
            'TEAM_ID': int(team_id),
            'TEAM_NAME': group['TEAM_NAME'].iloc[-1],
            'ELO_PRE': final_elos.get(int(team_id), 1500.0),
            'GAMES_PLAYED': len(group),
        }

        for feature in MODEL_SIGNAL_FEATURES:
            for window in ROLLING_WINDOWS:
                snapshot[f'R{window}_{feature}'] = group[feature].tail(window).mean()
            snapshot[f'STD_{feature}'] = group[feature].mean()
            home_group = group[group['IS_HOME'] == 1]
            away_group = group[group['IS_HOME'] == 0]
            snapshot[f'HOME_LOC_R{LOCATION_SPLIT_WINDOW}_{feature}'] = home_group[feature].tail(LOCATION_SPLIT_WINDOW).mean() if not home_group.empty else group[feature].tail(LOCATION_SPLIT_WINDOW).mean()
            snapshot[f'AWAY_LOC_R{LOCATION_SPLIT_WINDOW}_{feature}'] = away_group[feature].tail(LOCATION_SPLIT_WINDOW).mean() if not away_group.empty else group[feature].tail(LOCATION_SPLIT_WINDOW).mean()
            snapshot[f'HOME_LOC_STD_{feature}'] = home_group[feature].mean() if not home_group.empty else group[feature].mean()
            snapshot[f'AWAY_LOC_STD_{feature}'] = away_group[feature].mean() if not away_group.empty else group[feature].mean()

        last_game_date = group['GAME_DATE'].iloc[-1]
        rest_days = max(0, min((today - last_game_date).days, 7))
        recent_games = sum(1 for date in group['GAME_DATE'] if 0 < (today - date).days <= 3)
        snapshot['REST_DAYS'] = rest_days
        snapshot['B2B'] = 1 if rest_days <= 1 else 0
        snapshot['THREE_IN_FOUR'] = 1 if recent_games >= 2 else 0
        snapshots[int(team_id)] = snapshot

    return snapshots


def fetch_game_logs(seasons, league):
    all_games = []
    for season in seasons:
        games_season = leaguegamelog.LeagueGameLog(
            season=season,
            league_id=LEAGUES[league]['league_id']
        ).get_data_frames()[0]
        games_season['SEASON'] = season
        all_games.append(games_season)
        time.sleep(0.6)

    if not all_games:
        raise RuntimeError(f"No {league.upper()} games were available.")

    return pd.concat(all_games, ignore_index=True)


def calibrate_model(base_model, y):
    class_counts = y.value_counts()
    if len(class_counts) == 2 and class_counts.min() >= 3:
        try:
            return CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)
        except TypeError:
            return CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv=3)
    return base_model


def candidate_models(y):
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')
    )
    gradient_boosting = GradientBoostingClassifier(**MODEL_PARAMS)
    return {
        'logistic_regression': calibrate_model(logistic, y),
        'gradient_boosting': calibrate_model(gradient_boosting, y),
    }


def make_model(y, model_name='gradient_boosting'):
    return candidate_models(y)[model_name]


def calibration_curve_bins(y_true, probabilities, bins=5):
    frame = pd.DataFrame({'actual': y_true.reset_index(drop=True), 'probability': probabilities})
    frame['bin'] = pd.cut(frame['probability'], bins=bins, labels=False, include_lowest=True)
    return [
        {
            'bin': int(bin_id),
            'avg_probability': round(float(group['probability'].mean()), 4),
            'win_rate': round(float(group['actual'].mean()), 4),
            'games': int(len(group)),
        }
        for bin_id, group in frame.groupby('bin')
        if len(group) > 0
    ]


def validation_metrics(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        'accuracy': round(accuracy_score(y_test, predictions), 4),
        'log_loss': round(log_loss(y_test, probabilities, labels=[0, 1]), 4),
        'brier': round(brier_score_loss(y_test, probabilities), 4),
        'calibration': calibration_curve_bins(y_test, probabilities),
        'games': int(len(y_test)),
    }


def choose_model_by_time_validation(X_train, y_train, X_test, y_test):
    results = {}
    best_name = None
    best_model = None
    best_score = None

    for name, model in candidate_models(y_train).items():
        model.fit(X_train, y_train)
        metrics = validation_metrics(model, X_test, y_test)
        results[name] = metrics
        score = (metrics['log_loss'], metrics['brier'])
        if best_score is None or score < best_score:
            best_score = score
            best_name = name
            best_model = model

    return best_name, best_model, results


def ensure_model(league='nba'):
    league = normalize_league(league)
    if league in _MODEL_CACHE:
        return

    num_seasons = 10 if datetime.datetime.now().month in [4, 5, 6] else 5
    print(f"--- Training {league.upper()} pregame model ---")
    games = fetch_game_logs(get_seasons(num_seasons, league), league)
    X, y, dates = build_training_data(games)
    order = dates.sort_values().index
    X = X.loc[order].reset_index(drop=True)
    y = y.loc[order].reset_index(drop=True)

    selected_model_name = 'gradient_boosting'
    split_at = max(1, int(len(X) * 0.8))
    if split_at < len(X) and y.iloc[:split_at].nunique() == 2:
        selected_model_name, _, validation_results = choose_model_by_time_validation(
            X.iloc[:split_at],
            y.iloc[:split_at],
            X.iloc[split_at:],
            y.iloc[split_at:]
        )
        _METRICS_CACHE[league] = {
            'selected_model': selected_model_name,
            'candidates': validation_results,
        }
        print(f"--- {league.upper()} validation: {_METRICS_CACHE[league]} ---")

    model = make_model(y, selected_model_name)
    model.fit(X, y)
    _MODEL_CACHE[league] = model
    _FEATURES_CACHE[league] = X.columns.tolist()
    _SNAPSHOTS_CACHE[league] = build_team_snapshots(games, league)
    print(f"--- {league.upper()} Training Complete ---")


def is_model_ready(league='nba'):
    league = normalize_league(league)
    return league in _MODEL_CACHE and league in _FEATURES_CACHE


def train(team_id1, team_id2, league='nba'):
    league = normalize_league(league)
    ensure_model(league)
    if team_id1 is None or team_id2 is None:
        return None, None
    return predict_matchup(team_id1, team_id2, league)


def current_team_snapshots(league):
    league = normalize_league(league)
    if league in _SNAPSHOTS_CACHE:
        return _SNAPSHOTS_CACHE[league]

    seasons = get_seasons(1, league)
    games = fetch_game_logs(seasons, league)
    if games.empty and len(get_seasons(2, league)) > 1:
        games = fetch_game_logs([get_seasons(2, league)[0]], league)
    _SNAPSHOTS_CACHE[league] = build_team_snapshots(games, league)
    return _SNAPSHOTS_CACHE[league]


def location_aware_snapshot(snapshot, location):
    values = {}
    for feature in snapshot_feature_names():
        if feature.startswith('LOC_'):
            values[feature] = snapshot.get(f'{location}_{feature}', snapshot.get(feature, 0))
        else:
            values[feature] = snapshot.get(feature, 0)
    return values


def feature_row_from_snapshots(home_snapshot, away_snapshot, feature_columns):
    home_features = pd.DataFrame([location_aware_snapshot(home_snapshot, 'HOME')])
    away_features = pd.DataFrame([location_aware_snapshot(away_snapshot, 'AWAY')])
    row = matchup_feature_frame(home_features, away_features)
    for feature in feature_columns:
        if feature not in row.columns:
            row[feature] = 0
    return row[feature_columns]


def predict_matchup(team_id1, team_id2, league='nba'):
    league = normalize_league(league)
    if not is_model_ready(league):
        raise RuntimeError("Model is still training. Try again in a moment.")
    return match_up(_MODEL_CACHE[league], team_id1, team_id2, _FEATURES_CACHE[league], league)


def match_up(model, team1_id, team2_id, feature_columns, league='nba'):
    league = normalize_league(league)
    t1_id, t2_id = int(team1_id), int(team2_id)
    snapshots = current_team_snapshots(league)

    if t1_id not in snapshots or t2_id not in snapshots:
        return "Unknown", 0

    game_features = feature_row_from_snapshots(snapshots[t1_id], snapshots[t2_id], feature_columns)
    probability = model.predict_proba(game_features)[0]
    home_win_probability = probability[1]
    team1_name = snapshots[t1_id]['TEAM_NAME']
    team2_name = snapshots[t2_id]['TEAM_NAME']
    home_win_probability = injury_adjusted_home_probability(home_win_probability, team1_name, team2_name, league)
    winner = team1_name if home_win_probability >= 0.5 else team2_name
    confidence = max(home_win_probability, 1 - home_win_probability) * 100
    return winner, confidence


def period_seconds(league):
    return 600 if normalize_league(league) == 'wnba' else 720


def score_adjusted_projection(home_name, away_name, home_score, away_score, period, clock, model_home_prob, league, home_full_name=None, away_full_name=None):
    mins, secs, clock_match = parse_clock(clock)
    remaining_in_period = mins * 60 + secs
    periods_left = max(0, 4 - period)
    total_remaining = (periods_left * period_seconds(league)) + remaining_in_period
    regulation_seconds = 4 * period_seconds(league)
    time_factor = 1 - (total_remaining / regulation_seconds)

    margin = home_score - away_score
    sensitivity = 0.1 + (time_factor * 0.5)
    score_prob = 1 / (1 + math.exp(-margin * sensitivity))
    weight = 0.1 + (time_factor * 0.85)
    final_home_prob = (model_home_prob * (1 - weight)) + (score_prob * weight)

    winner = home_name if final_home_prob > 0.5 else away_name
    confidence = final_home_prob if final_home_prob > 0.5 else (1 - final_home_prob)
    return {
        'winner': winner,
        'confidence': round(confidence * 100, 2),
        'home': home_name,
        'away': away_name,
        'homeFullName': home_full_name or home_name,
        'awayFullName': away_full_name or away_name,
        'score': f"{away_score} - {home_score}",
        'clock': format_clock(clock) if clock_match else clock,
        'period': period
    }


def same_team_name(full_name, short_name):
    return full_name == short_name or short_name in full_name or full_name in short_name


def scoreboard_live_projection(game_id, league='nba'):
    league = normalize_league(league)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    games = scoreboardv3.ScoreboardV3(
        game_date=today,
        league_id=LEAGUES[league]['league_id']
    ).get_dict()['scoreboard']['games']
    game = next((game for game in games if game['gameId'] == game_id), None)
    if not game:
        raise Exception("Live scoreboard data was not found for this game.")

    home_team = game['homeTeam']
    away_team = game['awayTeam']
    home_name = home_team['teamName']
    away_name = away_team['teamName']
    home_full_name = full_team_name(home_team)
    away_full_name = full_team_name(away_team)
    home_score = home_team.get('score', 0)
    away_score = away_team.get('score', 0)
    period = game.get('period', 0)
    clock = game.get('gameClock', '')

    is_final = game.get('gameStatus') == 3 or game.get('gameStatusText', '').lower() == 'final'
    if is_final:
        return {
            'winner': home_name if home_score > away_score else away_name,
            'isFinal': True,
            'home': home_name,
            'away': away_name,
            'homeFullName': home_full_name,
            'awayFullName': away_full_name,
            'score': f"{away_score} - {home_score}",
            'clock': 'FINAL',
            'period': period
        }

    if game.get('gameStatus') == 1:
        raise Exception("Game has not started yet")

    if is_model_ready(league):
        model_winner, model_confidence = predict_matchup(home_team['teamId'], away_team['teamId'], league)
        model_home_prob = model_confidence / 100 if same_team_name(model_winner, home_name) else 1 - (model_confidence / 100)
    else:
        model_home_prob = 0.5
    return score_adjusted_projection(
        home_name,
        away_name,
        home_score,
        away_score,
        period,
        clock,
        model_home_prob,
        league,
        home_full_name,
        away_full_name
    )


def predict_live(game_id, league='nba'):
    league = normalize_league(league)
    base_url = LEAGUES[league]['live_boxscore_base_url']
    url = f"{base_url}/boxscore_{game_id}.json"
    res = requests.get(url, headers={'Referer': LEAGUES[league]['referer'], 'User-Agent': 'Mozilla/5.0'})
    if res.status_code != 200:
        return scoreboard_live_projection(game_id, league)

    try:
        box = res.json()['game']
    except ValueError:
        return scoreboard_live_projection(game_id, league)

    if box['gameStatus'] == 1:
        raise Exception("Game has not started yet")

    home_name = box['homeTeam']['teamName']
    away_name = box['awayTeam']['teamName']
    home_full_name = full_team_name(box['homeTeam'])
    away_full_name = full_team_name(box['awayTeam'])
    home_score = box['homeTeam']['score']
    away_score = box['awayTeam']['score']
    mins, secs, _ = parse_clock(box['gameClock'])
    is_clock_final = (mins == 0 and secs == 0 and box['period'] >= 4 and home_score != away_score)
    is_final = box['gameStatus'] == 3 or is_clock_final

    if is_final:
        return {
            'winner': home_name if home_score > away_score else away_name,
            'isFinal': True,
            'home': home_name,
            'away': away_name,
            'homeFullName': home_full_name,
            'awayFullName': away_full_name,
            'score': f"{away_score} - {home_score}",
            'clock': 'FINAL',
            'period': box['period']
        }

    if not is_model_ready(league):
        return score_adjusted_projection(
            home_name,
            away_name,
            home_score,
            away_score,
            box['period'],
            box['gameClock'],
            0.5,
            league,
            home_full_name,
            away_full_name
        )

    home_id = box['homeTeam'].get('teamId')
    away_id = box['awayTeam'].get('teamId')
    if home_id is None or away_id is None:
        return scoreboard_live_projection(game_id, league)

    model_winner, model_confidence = predict_matchup(home_id, away_id, league)
    model_home_prob = model_confidence / 100 if same_team_name(model_winner, home_name) else 1 - (model_confidence / 100)
    return score_adjusted_projection(
        home_name,
        away_name,
        home_score,
        away_score,
        box['period'],
        box['gameClock'],
        model_home_prob,
        league,
        home_full_name,
        away_full_name
    )