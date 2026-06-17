import pandas as pd
import requests
import re
from nba_api.stats.endpoints import leaguedashteamstats, leaguegamelog, scoreboardv3
from nba_api.stats.library.parameters import LeagueID, LeagueIDNullable
from sklearn.ensemble import RandomForestClassifier
import time
import datetime
import math


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
FEATURE_COLUMNS = ['OFF_RATING', 'TS_PCT', 'AST_100', 'REB_100', 'TOV_100', 'STL_100', 'BLK_100']
MODEL_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'max_depth': 12,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
}

_MODEL_CACHE = {}
_FEATURES_CACHE = {}


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

def add_rate_features(games):
    games = games.copy()
    games['POSS'] = games['FGA'] + 0.44 * games['FTA'] - games['OREB'] + games['TOV']
    games['OFF_RATING'] = (games['PTS'] / games['POSS']) * 100
    games['TS_PCT'] = games['PTS'] / (2 * (games['FGA'] + 0.44 * games['FTA']))
    games['AST_100'] = (games['AST'] / games['POSS']) * 100
    games['REB_100'] = (games['REB'] / games['POSS']) * 100
    games['TOV_100'] = (games['TOV'] / games['POSS']) * 100
    games['STL_100'] = (games['STL'] / games['POSS']) * 100
    games['BLK_100'] = (games['BLK'] / games['POSS']) * 100
    return games


def live_rate_features(team):
    stats = team['statistics']
    possessions = stats['fieldGoalsAttempted'] + 0.44 * stats['freeThrowsAttempted'] - stats['reboundsOffensive'] + stats['turnovers'] or 1
    shot_attempts = stats['fieldGoalsAttempted'] + 0.44 * stats['freeThrowsAttempted']

    return pd.Series({
        'OFF_RATING': (team['score'] / possessions) * 100,
        'TS_PCT': team['score'] / (2 * shot_attempts) if shot_attempts > 0 else 0,
        'AST_100': (stats['assists'] / possessions) * 100,
        'REB_100': (stats['reboundsTotal'] / possessions) * 100,
        'TOV_100': (stats['turnovers'] / possessions) * 100,
        'STL_100': (stats['steals'] / possessions) * 100,
        'BLK_100': (stats['blocks'] / possessions) * 100
    })


def matchup_features(team1_stats, team2_stats, feature_columns):
    stat_features = [feature for feature in feature_columns if feature != 'IS_HOME']
    game_diff = pd.DataFrame(team1_stats.values - team2_stats.values, columns=stat_features)
    game_diff['IS_HOME'] = 1
    return game_diff[feature_columns]


def parse_clock(clock):
    clock_match = re.search(r'PT(\d+)M(\d+(?:\.\d+)?)S?', clock or '')
    mins = int(clock_match.group(1)) if clock_match else 0
    secs = int(float(clock_match.group(2))) if clock_match else 0
    return mins, secs, clock_match


def format_clock(clock):
    mins, secs, clock_match = parse_clock(clock)
    return f"{mins}:{secs:02d}" if clock_match else clock


def build_training_data(games):
    games = add_rate_features(games)
    opp_stats = games[['GAME_ID', 'TEAM_ID'] + FEATURE_COLUMNS].copy()
    opp_stats.columns = ['GAME_ID', 'OPP_TEAM_ID'] + [f'OPP_{column}' for column in FEATURE_COLUMNS]

    merged = games.merge(opp_stats, on='GAME_ID')
    merged = merged[merged['TEAM_ID'] != merged['OPP_TEAM_ID']]
    merged['IS_HOME'] = merged['MATCHUP'].apply(lambda matchup: 1 if 'vs.' in matchup else 0)

    diff_data = pd.DataFrame({
        'WL': merged['WL'].map({'W': 1, 'L': 0}),
        'IS_HOME': merged['IS_HOME'],
    })

    for column in FEATURE_COLUMNS:
        diff_data[column] = merged[column] - merged[f'OPP_{column}']

    diff_data = diff_data.dropna()
    return diff_data.drop(columns=['WL']), diff_data['WL']


def ensure_model(league='nba'):
    league = normalize_league(league)

    if league in _MODEL_CACHE:
        return

    num_seasons = 10 if datetime.datetime.now().month in [4, 5, 6] else 5
    all_seasons_games = []

    print(f"--- Training {league.upper()} model ---")
    for season in get_seasons(num_seasons, league):
        try:
            games_season = leaguegamelog.LeagueGameLog(
                season=season,
                league_id=LEAGUES[league]['league_id']
            ).get_data_frames()[0]
            all_seasons_games.append(games_season)
            time.sleep(0.6)
        except:
            continue

    if not all_seasons_games:
        raise RuntimeError(f"No {league.upper()} games were available for training.")

    games = pd.concat(all_seasons_games, ignore_index=True)
    X, y = build_training_data(games)

    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X, y)
    _MODEL_CACHE[league] = model
    _FEATURES_CACHE[league] = X.columns.tolist()
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


def predict_matchup(team_id1, team_id2, league='nba'):
    league = normalize_league(league)
    if not is_model_ready(league):
        raise RuntimeError("Model is still training. Try again in a moment.")
    return match_up(_MODEL_CACHE[league], team_id1, team_id2, _FEATURES_CACHE[league], league)

def match_up(model, team1_id, team2_id, feature_columns, league='nba'):
    league = normalize_league(league)
    t1_id, t2_id = int(team1_id), int(team2_id)
    
    def get_stats(season_year=None):
        params = {
            'measure_type_detailed_defense': 'Base',
            'league_id_nullable': LEAGUES[league]['league_id_nullable'],
        }
        adv_params = {
            'measure_type_detailed_defense': 'Advanced',
            'league_id_nullable': LEAGUES[league]['league_id_nullable'],
        }
        if season_year:
            params['season'] = season_year
            adv_params['season'] = season_year
            
        base = leaguedashteamstats.LeagueDashTeamStats(**params).get_data_frames()[0]
        adv = leaguedashteamstats.LeagueDashTeamStats(**adv_params).get_data_frames()[0]
        return pd.merge(base, adv, on=['TEAM_ID', 'TEAM_NAME'], suffixes=('', '_adv'))

    current_season = get_seasons(1, league)[0]
    stats = get_stats(season_year=current_season)

    if stats.empty:
        last_season = get_seasons(2, league)[0]
        stats = get_stats(season_year=last_season)
 
    stats = add_rate_features(stats)
    
    stat_features = [f for f in feature_columns if f != 'IS_HOME']
    team1_stats = stats[stats['TEAM_ID'] == t1_id][stat_features]
    team2_stats = stats[stats['TEAM_ID'] == t2_id][stat_features]
    
    if team1_stats.empty or team2_stats.empty:
        return "Unknown", 0

    team1_name = stats[stats['TEAM_ID'] == t1_id]['TEAM_NAME'].values[0]
    team2_name = stats[stats['TEAM_ID'] == t2_id]['TEAM_NAME'].values[0]

    print(f"\n--- PRE-GAME STATS ---")
    print(f"{team1_name}:\n{team1_stats.iloc[0].to_dict()}")
    print(f"{team2_name}:\n{team2_stats.iloc[0].to_dict()}")
    
    game_diff = matchup_features(team1_stats, team2_stats, feature_columns)
    
    prediction = model.predict(game_diff)
    probability = model.predict_proba(game_diff)[0]

    team1_name = stats[stats['TEAM_ID'] == t1_id]['TEAM_NAME'].values[0]
    team2_name = stats[stats['TEAM_ID'] == t2_id]['TEAM_NAME'].values[0]

    winner = team1_name if prediction[0] == 1 else team2_name
    confidence = max(probability) * 100

    return winner, confidence


def period_seconds(league):
    return 600 if normalize_league(league) == 'wnba' else 720


def score_adjusted_projection(home_name, away_name, home_score, away_score, period, clock, model_home_prob, league):
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
            'score': f"{away_score} - {home_score}",
            'clock': 'FINAL',
            'period': period
        }

    if game.get('gameStatus') == 1:
        raise Exception("Game has not started yet")

    if not is_model_ready(league):
        raise RuntimeError("Model is still training. Try again in a moment.")

    model_winner, model_confidence = predict_matchup(home_team['teamId'], away_team['teamId'], league)
    model_home_prob = model_confidence / 100 if same_team_name(model_winner, home_name) else 1 - (model_confidence / 100)

    return score_adjusted_projection(
        home_name,
        away_name,
        home_score,
        away_score,
        period,
        clock,
        model_home_prob,
        league
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
    
    if box['gameStatus'] == 1: raise Exception("Game has not started yet")

    home_name = box['homeTeam']['teamName']
    away_name = box['awayTeam']['teamName']
    home_score = box['homeTeam']['score']
    away_score = box['awayTeam']['score']

    # Treat as final if official status is 3 OR if clock is 0:00 in 4th+ and not tied
    mins, secs, clock_match = parse_clock(box['gameClock'])

    is_clock_final = (mins == 0 and secs == 0 and box['period'] >= 4 and home_score != away_score)
    is_final = box['gameStatus'] == 3 or is_clock_final

    if is_final:
        return {
            'winner': home_name if home_score > away_score else away_name,
            'isFinal': True,
            'home': home_name,
            'away': away_name,
            'score': f"{away_score} - {home_score}",
            'clock': 'FINAL',
            'period': box['period']
        }

    if not is_model_ready(league):
        raise RuntimeError("Model is still training. Try again in a moment.")

    h_stats = live_rate_features(box['homeTeam'])
    a_stats = live_rate_features(box['awayTeam'])
    
    # print(f"\n--- LIVE STATS ({home_name} vs {away_name}) ---")
    # print(f"{home_name} (Home):\n{h_stats.to_dict()}")
    # print(f"{away_name} (Away):\n{a_stats.to_dict()}")

    diff = matchup_features(h_stats.to_frame().T, a_stats.to_frame().T, _FEATURES_CACHE[league])
    
    ml_prob = _MODEL_CACHE[league].predict_proba(diff)[0]

    return score_adjusted_projection(
        home_name,
        away_name,
        home_score,
        away_score,
        box['period'],
        box['gameClock'],
        ml_prob[1],
        league
    )
