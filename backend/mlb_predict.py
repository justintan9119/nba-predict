import datetime
import math
import os
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MLB_SPORT_ID = 1
TEAM_SIDES = ('away', 'home')
TEAM_LOGO_URL = 'https://www.mlbstatic.com/team-logos/{team_id}.svg'
PROFILE_FEATURES = [
    'runs_per_game',
    'ops',
    'obp',
    'slg',
    'walk_rate',
    'strikeout_rate',
    'era',
    'whip',
    'runs_allowed_per_game',
    'strikeout_walk_ratio',
    'record_pct',
]
ROLLING_WINDOWS = (7, 14, 30)
CONTEXT_FEATURES = [
    'STARTER_ERA',
    'STARTER_WHIP',
    'STARTER_K_BB',
    'STARTER_IP_PER_START',
    'LINEUP_OPS',
    'LINEUP_OBP',
    'LINEUP_SLG',
    'BULLPEN_ERA_7',
    'BULLPEN_WHIP_7',
    'BULLPEN_FATIGUE_3',
]
GAME_CONTEXT_FEATURES = [
    'PARK_FACTOR',
    'TEMP_F',
    'WIND_OUT_MPH',
    'MARKET_HOME_IMPLIED',
]
PREGAME_FEATURES = (
    [f'DIFF_{feature}' for feature in PROFILE_FEATURES]
    + [f'DIFF_{feature}_LAST_{window}' for window in ROLLING_WINDOWS for feature in PROFILE_FEATURES]
    + [f'DIFF_{feature}' for feature in CONTEXT_FEATURES]
    + GAME_CONTEXT_FEATURES
    + ['HOME_RECORD_PCT', 'AWAY_RECORD_PCT']
)
LIVE_FEATURES = [
    'PREGAME_HOME_PROB',
    'PROGRESS',
    'INNING',
    'OUTS',
    'SCORE_DIFF',
    'HIT_DIFF',
    'TOTAL_BASE_DIFF',
    'HOME_RUN_DIFF',
    'WALK_DIFF',
    'BATTING_K_EDGE',
    'PITCHING_K_DIFF',
    'ERROR_EDGE',
    'LOB_EDGE',
]
MODEL_CACHE_VERSION = 6
MODEL_CACHE_DIR = Path(os.environ.get('MLB_MODEL_CACHE_DIR', Path(__file__).with_name('model_cache')))
MLB_TRAINING_SEASONS = max(2, int(os.environ.get('MLB_TRAINING_SEASONS', '2')))
MLB_USE_HISTORICAL_BOX = os.environ.get('MLB_USE_HISTORICAL_BOX', '0') == '1'
MLB_LIVE_TRAINING_GAMES = max(0, int(os.environ.get('MLB_LIVE_TRAINING_GAMES', '0')))
MLB_USE_MARKET_FEATURE = os.environ.get('MLB_USE_MARKET_FEATURE', '0') == '1'
LIVE_PROGRESS_CHECKPOINTS = (0.2, 0.45, 0.7, 0.9)
DEFAULT_PARK_FACTOR = 1.0
PARK_FACTORS = {
    2: 0.98, 3: 1.05, 4: 1.01, 5: 0.97, 7: 1.02, 10: 1.00, 12: 0.99, 15: 1.04,
    17: 1.03, 19: 1.01, 22: 0.96, 31: 1.02, 32: 1.00, 2392: 0.97, 2394: 0.99,
    2395: 1.00, 2397: 1.00, 2399: 0.98, 2680: 1.02, 3289: 0.96, 3309: 1.01,
    3312: 1.03, 3313: 1.00, 3316: 1.01, 3317: 1.00, 3319: 0.99, 5325: 0.95,
}

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE = None
_TRAINING = False
_METRICS_CACHE = None


def require_mlb_statsapi():
    try:
        import mlbstatsapi
    except ImportError as error:
        raise RuntimeError(
            'MLB support requires python-mlb-statsapi. Install it in the backend environment with '
            'pip install python-mlb-statsapi.'
        ) from error
    return mlbstatsapi


@lru_cache(maxsize=1)
def mlb_client():
    return require_mlb_statsapi().Mlb()


def dump_model(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return value
    return value


def get_path(value, *path, default=None):
    current = dump_model(value)
    for part in path:
        current = dump_model(current)
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return default
    return dump_model(current)


def parse_number(value, default=0.0):
    if value in (None, '', '-.--'):
        return default
    try:
        return float(str(value).replace('%', ''))
    except (TypeError, ValueError):
        return default


def int_or_zero(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def selected_mlb_season():
    today = datetime.date.today()
    return today.year if today.month >= 3 else today.year - 1


def mlb_season_for_date(value):
    try:
        selected = datetime.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return selected_mlb_season()
    return selected.year if selected.month >= 3 else selected.year - 1


def ordinal_inning(inning):
    inning = int_or_zero(inning)
    if inning <= 0:
        return ''
    suffix = 'th'
    if inning % 100 not in {11, 12, 13}:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(inning % 10, 'th')
    return f'{inning}{suffix}'


def short_team_name(name):
    parts = str(name or '').split()
    return parts[-1] if parts else ''


def schedule_dates(schedule):
    data = dump_model(schedule)
    return data.get('dates', []) if isinstance(data, dict) else []


def game_team(game, side):
    return get_path(game, 'teams', side, 'team', default={}) or {}


def game_team_id(game, side):
    return int_or_zero(game_team(game, side).get('id'))


def box_team(box_score, side):
    return get_path(box_score, 'teams', side, default={}) or {}


def box_team_info(box_score, side):
    return get_path(box_score, 'teams', side, 'team', default={}) or {}


def team_record_pct(source, side):
    team = game_team(source, side) or box_team_info(source, side)
    return (
        get_path(source, 'teams', side, 'league_record', 'pct')
        or get_path(team, 'record', 'league_record', 'pct')
        or get_path(team, 'record', 'winning_percentage')
        or 0.5
    )


def game_state(game):
    detailed = get_path(game, 'status', 'detailed_state', default='') or ''
    abstract = get_path(game, 'status', 'abstract_game_state', default='') or ''
    detailed_lower = detailed.lower()
    abstract_lower = abstract.lower()
    if abstract_lower == 'final' or 'final' in detailed_lower or 'completed' in detailed_lower:
        return 3, True, False
    if abstract_lower == 'live' or 'in progress' in detailed_lower or 'warmup' in detailed_lower:
        return 2, False, True
    return 1, False, False


def line_score_text(game):
    inning = get_path(game, 'linescore', 'current_inning', default=0)
    inning_label = get_path(game, 'linescore', 'current_inning_ordinal', default='') or ordinal_inning(inning)
    half = get_path(game, 'linescore', 'inning_half', default='') or get_path(game, 'linescore', 'inning_state', default='')
    if inning_label and half:
        return f'{half} {inning_label}'
    return get_path(game, 'status', 'detailed_state', default='') or 'Scheduled'


def mlb_game_payload(game):
    status, is_final, is_live = game_state(game)
    away_team = game_team(game, 'away')
    home_team = game_team(game, 'home')
    away_id = game_team_id(game, 'away')
    home_id = game_team_id(game, 'home')
    return {
        'gameId': str(get_path(game, 'game_pk', default='')),
        'away': short_team_name(away_team.get('name')),
        'home': short_team_name(home_team.get('name')),
        'awayFullName': away_team.get('name'),
        'homeFullName': home_team.get('name'),
        'awayTeamId': away_id,
        'homeTeamId': home_id,
        'awayLogo': TEAM_LOGO_URL.format(team_id=away_id),
        'homeLogo': TEAM_LOGO_URL.format(team_id=home_id),
        'awayScore': int_or_zero(get_path(game, 'teams', 'away', 'score', default=0)),
        'homeScore': int_or_zero(get_path(game, 'teams', 'home', 'score', default=0)),
        'status': status,
        'statusText': get_path(game, 'status', 'detailed_state', default='Scheduled'),
        'startTime': get_path(game, 'game_date', default=None),
        'clock': line_score_text(game),
        'period': int_or_zero(get_path(game, 'linescore', 'current_inning', default=0)),
        'isLive': is_live,
        'isFinal': is_final,
    }


def mlb_scoreboard_games(selected_date):
    schedule = mlb_client().get_schedule(date=selected_date, sport_id=MLB_SPORT_ID)
    games = []
    for schedule_date in schedule_dates(schedule):
        games.extend(mlb_game_payload(game) for game in schedule_date.get('games', []))
    return games


def find_mlb_schedule_game(game_id, selected_date):
    schedule = mlb_client().get_schedule(date=selected_date, sport_id=MLB_SPORT_ID)
    for schedule_date in schedule_dates(schedule):
        for game in schedule_date.get('games', []):
            if str(get_path(game, 'game_pk', default='')) == str(game_id):
                return game
    return None


def first_split_stat(stat_group):
    stat_group = dump_model(stat_group)
    splits = stat_group.get('splits', []) if isinstance(stat_group, dict) else []
    if not splits:
        return {}
    return get_path(splits[0], 'stat', default={}) or {}


def stat_bucket(stats, group, stat_type):
    stats = dump_model(stats)
    if not isinstance(stats, dict):
        return {}
    group_stats = stats.get(group, {})
    if not isinstance(group_stats, dict):
        return {}
    return first_split_stat(group_stats.get(stat_type) or group_stats.get(stat_type.lower()) or {})


@lru_cache(maxsize=512)
def raw_team_stats(team_id, season):
    stats = ['season', 'seasonAdvanced']
    groups = ['hitting', 'pitching']
    return mlb_client().get_team_stats(int(team_id), stats=stats, groups=groups, season=season)


@lru_cache(maxsize=4096)
def raw_game_box_score(game_id):
    return dump_model(mlb_client().get_game_box_score(int(game_id)))


@lru_cache(maxsize=4096)
def raw_game_play_by_play(game_id):
    return dump_model(mlb_client().get_game_play_by_play(int(game_id)))


def per_game(total, games):
    games = max(parse_number(games), 1)
    return parse_number(total) / games


def safe_div(numerator, denominator, default=0.0):
    denominator = parse_number(denominator)
    if denominator == 0:
        return default
    return parse_number(numerator) / denominator


def innings_to_outs(innings):
    if innings in (None, '', '-.--', '-'):
        return 0
    text = str(innings)
    if '.' not in text:
        return int_or_zero(text) * 3
    whole, fraction = text.split('.', 1)
    return int_or_zero(whole) * 3 + min(int_or_zero(fraction[:1]), 2)


def pitcher_totals_from_box(box_score, side):
    team = box_team(box_score, side)
    players = team.get('players', {}) or {}
    pitcher_ids = [f'ID{player_id}' for player_id in team.get('pitchers', [])]
    relief = {
        'relief_outs': 0,
        'relief_hits': 0,
        'relief_walks': 0,
        'relief_earned_runs': 0,
        'relief_pitches': 0,
    }
    for index, player_id in enumerate(pitcher_ids):
        player = players.get(player_id) or {}
        pitching = (player.get('stats', {}) or {}).get('pitching', {}) or {}
        if index == 0:
            continue
        relief['relief_outs'] += parse_number(pitching.get('outs'), innings_to_outs(pitching.get('inningsPitched')))
        relief['relief_hits'] += parse_number(pitching.get('hits'))
        relief['relief_walks'] += parse_number(pitching.get('baseOnBalls'))
        relief['relief_earned_runs'] += parse_number(pitching.get('earnedRuns'))
        relief['relief_pitches'] += parse_number(pitching.get('numberOfPitches') or pitching.get('pitchesThrown'))
    return relief


def empty_team_totals():
    return {
        'games': 0,
        'wins': 0,
        'runs': 0,
        'runs_allowed': 0,
        'hits': 0,
        'doubles': 0,
        'triples': 0,
        'home_runs': 0,
        'walks': 0,
        'strikeouts': 0,
        'plate_appearances': 0,
        'at_bats': 0,
        'total_bases': 0,
        'pitching_hits': 0,
        'pitching_walks': 0,
        'pitching_strikeouts': 0,
        'pitching_outs': 0,
        'earned_runs': 0,
        'relief_outs': 0,
        'relief_hits': 0,
        'relief_walks': 0,
        'relief_earned_runs': 0,
        'relief_pitches': 0,
    }


def add_totals(left, right):
    for key, value in right.items():
        left[key] += value
    return left


def totals_from_recent_games(games):
    totals = empty_team_totals()
    for game_totals in games:
        add_totals(totals, game_totals)
    return totals


def profile_with_rolling_windows(season_totals, recent_games):
    profile = team_profile_from_totals(season_totals)
    for window in ROLLING_WINDOWS:
        rolling_profile = team_profile_from_totals(totals_from_recent_games(recent_games[-window:]))
        for feature in PROFILE_FEATURES:
            profile[f'{feature}_last_{window}'] = rolling_profile.get(feature, team_profile_from_totals(empty_team_totals())[feature])
        profile[f'bullpen_era_last_{window}'] = rolling_profile.get('bullpen_era', 4.25)
        profile[f'bullpen_whip_last_{window}'] = rolling_profile.get('bullpen_whip', 1.30)
        profile[f'bullpen_fatigue_last_{window}'] = rolling_profile.get('bullpen_fatigue', 0.0)
    return profile


def team_profile_from_totals(totals):
    games = int(totals.get('games', 0))
    if games <= 0:
        return {
            'runs_per_game': 4.5,
            'ops': 0.710,
            'obp': 0.315,
            'slg': 0.395,
            'walk_rate': 0.085,
            'strikeout_rate': 0.225,
            'era': 4.25,
            'whip': 1.30,
            'runs_allowed_per_game': 4.5,
            'strikeout_walk_ratio': 2.35,
            'record_pct': 0.5,
            'bullpen_era': 4.25,
            'bullpen_whip': 1.30,
            'bullpen_fatigue': 0.0,
        }

    raw_at_bats = parse_number(totals.get('at_bats'))
    raw_plate_appearances = parse_number(totals.get('plate_appearances'))
    at_bats = max(raw_at_bats, 1)
    plate_appearances = max(raw_plate_appearances, 1)
    pitching_outs = max(parse_number(totals.get('pitching_outs')), 1)
    pitching_innings = pitching_outs / 3
    obp = (
        safe_div(totals.get('hits') + totals.get('walks'), plate_appearances, 0.315)
        if raw_plate_appearances > 0
        else 0.315
    )
    slg = safe_div(totals.get('total_bases'), at_bats, 0.395) if raw_at_bats > 0 else 0.395
    walk_rate = safe_div(totals.get('walks'), plate_appearances, 0.085) if raw_plate_appearances > 0 else 0.085
    strikeout_rate = safe_div(totals.get('strikeouts'), plate_appearances, 0.225) if raw_plate_appearances > 0 else 0.225
    whip = (
        (parse_number(totals.get('pitching_hits')) + parse_number(totals.get('pitching_walks'))) / pitching_innings
        if parse_number(totals.get('pitching_hits')) or parse_number(totals.get('pitching_walks'))
        else 1.30
    )
    relief_outs = max(parse_number(totals.get('relief_outs')), 1)
    relief_innings = relief_outs / 3
    relief_whip = (
        (parse_number(totals.get('relief_hits')) + parse_number(totals.get('relief_walks'))) / relief_innings
        if parse_number(totals.get('relief_hits')) or parse_number(totals.get('relief_walks'))
        else 1.30
    )

    return {
        'runs_per_game': parse_number(totals.get('runs')) / games,
        'ops': obp + slg,
        'obp': obp,
        'slg': slg,
        'walk_rate': walk_rate,
        'strikeout_rate': strikeout_rate,
        'era': parse_number(totals.get('earned_runs')) * 9 / pitching_innings,
        'whip': whip,
        'runs_allowed_per_game': parse_number(totals.get('runs_allowed')) / games,
        'strikeout_walk_ratio': safe_div(totals.get('pitching_strikeouts'), totals.get('pitching_walks'), 2.35),
        'record_pct': safe_div(totals.get('wins'), games, 0.5),
        'bullpen_era': parse_number(totals.get('relief_earned_runs')) * 9 / relief_innings,
        'bullpen_whip': relief_whip,
        'bullpen_fatigue': parse_number(totals.get('relief_pitches')),
    }


def team_game_totals(box_score, side, won):
    team_stats = get_path(box_score, 'teams', side, 'team_stats', default={}) or {}
    batting = team_stats.get('batting', {})
    pitching = team_stats.get('pitching', {})
    hits = parse_number(batting.get('hits'))
    doubles = parse_number(batting.get('doubles'))
    triples = parse_number(batting.get('triples'))
    home_runs = parse_number(batting.get('homeRuns'))
    total_bases = parse_number(batting.get('totalBases'), hits + doubles + (2 * triples) + (3 * home_runs))
    return {
        'games': 1,
        'wins': 1 if won else 0,
        'runs': parse_number(batting.get('runs')),
        'runs_allowed': parse_number(pitching.get('runs')),
        'hits': hits,
        'doubles': doubles,
        'triples': triples,
        'home_runs': home_runs,
        'walks': parse_number(batting.get('baseOnBalls')),
        'strikeouts': parse_number(batting.get('strikeOuts')),
        'plate_appearances': parse_number(batting.get('plateAppearances')),
        'at_bats': parse_number(batting.get('atBats')),
        'total_bases': total_bases,
        'pitching_hits': parse_number(pitching.get('hits')),
        'pitching_walks': parse_number(pitching.get('baseOnBalls')),
        'pitching_strikeouts': parse_number(pitching.get('strikeOuts')),
        'pitching_outs': parse_number(pitching.get('outs'), innings_to_outs(pitching.get('inningsPitched'))),
        'earned_runs': parse_number(pitching.get('earnedRuns')),
        **pitcher_totals_from_box(box_score, side),
    }


def fallback_game_totals(game, side, won):
    opponent = 'away' if side == 'home' else 'home'
    runs = game_score(game, side)
    runs_allowed = game_score(game, opponent)
    return {
        **empty_team_totals(),
        'games': 1,
        'wins': 1 if won else 0,
        'runs': runs,
        'runs_allowed': runs_allowed,
        'earned_runs': runs_allowed,
        'pitching_outs': 27,
    }


def team_metric_profile(team_id, season, record_pct=0.5):
    stats = raw_team_stats(int(team_id), int(season))
    hitting = stat_bucket(stats, 'hitting', 'season')
    advanced_hitting = stat_bucket(stats, 'hitting', 'seasonAdvanced')
    pitching = stat_bucket(stats, 'pitching', 'season')
    advanced_pitching = stat_bucket(stats, 'pitching', 'seasonAdvanced')

    hitting_games = hitting.get('games_played') or hitting.get('gamesPlayed')
    pitching_games = pitching.get('games_played') or pitching.get('gamesPlayed')
    walks = hitting.get('base_on_balls') or hitting.get('baseOnBalls')
    strikeouts = hitting.get('strike_outs') or hitting.get('strikeOuts')
    pitching_walks = pitching.get('base_on_balls') or pitching.get('baseOnBalls')
    pitching_strikeouts = pitching.get('strike_outs') or pitching.get('strikeOuts')

    return {
        'runs_per_game': per_game(hitting.get('runs'), hitting_games),
        'ops': parse_number(hitting.get('ops') or advanced_hitting.get('ops'), 0.700),
        'obp': parse_number(hitting.get('obp') or advanced_hitting.get('obp'), 0.315),
        'slg': parse_number(hitting.get('slg') or advanced_hitting.get('slg'), 0.390),
        'walk_rate': parse_number(advanced_hitting.get('walk_percentage') or advanced_hitting.get('walkPercentage'), safe_div(walks, hitting.get('plate_appearances'))),
        'strikeout_rate': parse_number(advanced_hitting.get('strikeout_percentage') or advanced_hitting.get('strikeoutPercentage'), safe_div(strikeouts, hitting.get('plate_appearances'))),
        'era': parse_number(pitching.get('era'), 4.25),
        'whip': parse_number(pitching.get('whip') or advanced_pitching.get('whip'), 1.30),
        'runs_allowed_per_game': per_game(pitching.get('runs'), pitching_games),
        'strikeout_walk_ratio': parse_number(
            pitching.get('strikeout_walk_ratio') or pitching.get('strikeoutWalkRatio') or advanced_pitching.get('strikeout_walk_ratio'),
            safe_div(pitching_strikeouts, pitching_walks, 2.2),
        ),
        'record_pct': parse_number(record_pct, 0.5),
    }


def mlb_training_seasons(count=MLB_TRAINING_SEASONS):
    current = selected_mlb_season()
    return list(range(current - count, current + 1))


def model_cache_path(seasons):
    season_key = '-'.join(str(season) for season in seasons)
    return MODEL_CACHE_DIR / f'mlb-sklearn-{season_key}-v{MODEL_CACHE_VERSION}.joblib'


def load_cached_model(seasons):
    global _MODEL_CACHE, _METRICS_CACHE
    path = model_cache_path(seasons)
    if not path.exists():
        return False
    try:
        payload = joblib.load(path)
    except (OSError, ValueError, EOFError):
        return False
    if payload.get('version') != MODEL_CACHE_VERSION or payload.get('seasons') != seasons:
        return False
    _MODEL_CACHE = payload
    _METRICS_CACHE = payload.get('metrics')
    print(f"--- Loaded MLB sklearn model cache: {path.name} ---")
    return True


def save_model_cache(payload, seasons):
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = model_cache_path(seasons)
    temporary_path = path.with_suffix('.tmp')
    joblib.dump(payload, temporary_path)
    temporary_path.replace(path)


def weather_value(game, *keys, default=0.0):
    weather = get_path(game, 'weather', default={}) or {}
    for key in keys:
        value = get_path(weather, key, default=None)
        if value not in (None, ''):
            return parse_number(value, default)
        value = get_path(game, key, default=None)
        if value not in (None, ''):
            return parse_number(value, default)
    return default


def game_park_factor(game):
    venue_id = int_or_zero(get_path(game, 'venue', 'id', default=0))
    return PARK_FACTORS.get(venue_id, DEFAULT_PARK_FACTOR)


def wind_out_mph(game):
    weather = get_path(game, 'weather', default={}) or {}
    wind_text = str(weather.get('wind') or get_path(game, 'wind', default='') or '').lower()
    speed = parse_number(wind_text.split()[0] if wind_text else None, 0.0)
    if not speed:
        return 0.0
    if 'out' in wind_text or 'to center' in wind_text or 'to left' in wind_text or 'to right' in wind_text:
        return speed
    if 'in' in wind_text or 'from center' in wind_text or 'from left' in wind_text or 'from right' in wind_text:
        return -speed
    return 0.0


def market_home_implied_probability(game):
    if not MLB_USE_MARKET_FEATURE:
        return 0.5
    try:
        from odds import fetch_moneyline_odds

        odds = fetch_moneyline_odds('mlb', mlb_game_payload(game))
    except Exception:
        return 0.5
    if not odds:
        return 0.5
    home_price = odds.get('home', {}).get('price')
    away_price = odds.get('away', {}).get('price')
    home_raw = american_implied_probability(home_price)
    away_raw = american_implied_probability(away_price)
    total = home_raw + away_raw
    return home_raw / total if total > 0 else 0.5


def american_implied_probability(value):
    value = parse_number(value)
    if value > 0:
        return 100 / (value + 100)
    if value < 0:
        return abs(value) / (abs(value) + 100)
    return 0.5


def game_context_features(game, market_home_probability=None):
    return {
        'PARK_FACTOR': game_park_factor(game),
        'TEMP_F': weather_value(game, 'temp', 'temperature', default=72.0),
        'WIND_OUT_MPH': wind_out_mph(game),
        'MARKET_HOME_IMPLIED': market_home_probability if market_home_probability is not None else 0.5,
    }


def pregame_feature_row(home_profile, away_profile, home_context=None, away_context=None, game_context=None):
    row = {}
    home_profile = with_profile_defaults(home_profile)
    away_profile = with_profile_defaults(away_profile)
    home_context = home_context or {}
    away_context = away_context or {}
    game_context = game_context or {}
    for feature in PROFILE_FEATURES:
        row[f'DIFF_{feature}'] = parse_number(home_profile.get(feature)) - parse_number(away_profile.get(feature))
    for window in ROLLING_WINDOWS:
        for feature in PROFILE_FEATURES:
            key = f'{feature}_last_{window}'
            row[f'DIFF_{feature}_LAST_{window}'] = parse_number(home_profile.get(key)) - parse_number(away_profile.get(key))
    for feature in CONTEXT_FEATURES:
        row[f'DIFF_{feature}'] = parse_number(home_context.get(feature), context_default(feature)) - parse_number(away_context.get(feature), context_default(feature))
    for feature in GAME_CONTEXT_FEATURES:
        row[feature] = parse_number(game_context.get(feature), game_context_default(feature))
    row['HOME_RECORD_PCT'] = parse_number(home_profile.get('record_pct'), 0.5)
    row['AWAY_RECORD_PCT'] = parse_number(away_profile.get('record_pct'), 0.5)
    return row


def context_default(feature):
    return {
        'STARTER_ERA': 4.25,
        'STARTER_WHIP': 1.30,
        'STARTER_K_BB': 2.35,
        'STARTER_IP_PER_START': 5.2,
        'LINEUP_OPS': 0.710,
        'LINEUP_OBP': 0.315,
        'LINEUP_SLG': 0.395,
        'BULLPEN_ERA_7': 4.25,
        'BULLPEN_WHIP_7': 1.30,
        'BULLPEN_FATIGUE_3': 0.0,
    }.get(feature, 0.0)


def game_context_default(feature):
    return {
        'PARK_FACTOR': DEFAULT_PARK_FACTOR,
        'TEMP_F': 72.0,
        'WIND_OUT_MPH': 0.0,
        'MARKET_HOME_IMPLIED': 0.5,
    }.get(feature, 0.0)


def schedule_regular_games(season):
    schedule = mlb_client().get_schedule(
        start_date=f'{season}-03-01',
        end_date=f'{season}-11-30',
        sport_id=MLB_SPORT_ID,
    )
    for schedule_date in schedule_dates(schedule):
        for game in schedule_date.get('games', []):
            if str(game.get('game_type') or game.get('gameType') or '').upper() != 'R':
                continue
            status, is_final, _ = game_state(game)
            if status == 3 or is_final:
                yield game


def game_sort_key(game):
    return (
        get_path(game, 'official_date', default='') or '',
        get_path(game, 'game_date', default='') or '',
        int_or_zero(get_path(game, 'game_pk', default=0)),
    )


def sorted_regular_games(season):
    return sorted(schedule_regular_games(season), key=game_sort_key)


def game_score(game, side):
    return int_or_zero(get_path(game, 'teams', side, 'score', default=0))


def update_team_totals_from_game(totals_by_team, game, history_by_team=None):
    game_id = get_path(game, 'game_pk')
    home_id = game_team_id(game, 'home')
    away_id = game_team_id(game, 'away')
    if not home_id or not away_id:
        return None

    home_score = game_score(game, 'home')
    away_score = game_score(game, 'away')
    if home_score == away_score:
        return None

    try:
        if not MLB_USE_HISTORICAL_BOX:
            raise RuntimeError('Historical box-score backfill disabled')
        box_score = raw_game_box_score(int(game_id))
        if not box_score:
            raise RuntimeError('MLB box score unavailable')
        home_totals = team_game_totals(box_score, 'home', home_score > away_score)
        away_totals = team_game_totals(box_score, 'away', away_score > home_score)
    except Exception:
        home_totals = fallback_game_totals(game, 'home', home_score > away_score)
        away_totals = fallback_game_totals(game, 'away', away_score > home_score)

    add_totals(totals_by_team[home_id], home_totals)
    add_totals(totals_by_team[away_id], away_totals)
    if history_by_team is not None:
        history_by_team[home_id].append(home_totals)
        history_by_team[away_id].append(away_totals)
    return {'home': home_totals, 'away': away_totals}


@lru_cache(maxsize=256)
def team_profiles_before_date(season, cutoff_date):
    totals_by_team = defaultdict(empty_team_totals)
    history_by_team = defaultdict(list)
    for game in sorted_regular_games(int(season)):
        game_date = get_path(game, 'official_date', default='') or ''
        if game_date >= cutoff_date:
            break
        update_team_totals_from_game(totals_by_team, game, history_by_team)
    return {
        int(team_id): profile_with_rolling_windows(totals, history_by_team[int(team_id)])
        for team_id, totals in totals_by_team.items()
    }


def team_profile_for_game(game, side, season=None, profiles_by_team=None):
    team = game_team(game, side)
    if not team.get('id'):
        return None
    if profiles_by_team is not None:
        return profiles_by_team.get(int(team.get('id')), profile_with_rolling_windows(empty_team_totals(), []))
    return with_profile_defaults(team_metric_profile(
        team.get('id'),
        season or selected_mlb_season(),
        team_record_pct(game, side),
    ))


def with_profile_defaults(profile):
    neutral = profile_with_rolling_windows(empty_team_totals(), [])
    merged = dict(neutral)
    merged.update(profile or {})
    return merged


def player_stat_profile(player):
    player = dump_model(player) or {}
    stats = player.get('season_stats') or player.get('stats') or {}
    pitching = stats.get('pitching', {}) or {}
    batting = stats.get('batting', {}) or {}
    return batting, pitching


def pitcher_quality_from_player(player):
    _, pitching = player_stat_profile(player)
    innings = parse_number(pitching.get('inningsPitched'))
    games_started = max(parse_number(pitching.get('gamesStarted') or pitching.get('games_started')), 1)
    walks = parse_number(pitching.get('baseOnBalls') or pitching.get('walks'))
    return {
        'STARTER_ERA': parse_number(pitching.get('era'), 4.25),
        'STARTER_WHIP': parse_number(pitching.get('whip'), 1.30),
        'STARTER_K_BB': safe_div(pitching.get('strikeOuts') or pitching.get('strikeouts'), walks, 2.35),
        'STARTER_IP_PER_START': innings / games_started if innings > 0 else 5.2,
    }


def probable_pitcher_id(game, side):
    pitcher = (
        get_path(game, 'teams', side, 'probable_pitcher', default={})
        or get_path(game, 'teams', side, 'probablePitcher', default={})
        or {}
    )
    return int_or_zero(pitcher.get('id'))


def box_player_by_person_id(box_score, side, person_id):
    if not box_score or not person_id:
        return None
    players = box_team(box_score, side).get('players', {}) or {}
    return next(
        (player for player in players.values() if int_or_zero(get_path(player, 'person', 'id', default=0)) == int(person_id)),
        None,
    )


def starter_context(game, box_score, side):
    player = box_player_by_person_id(box_score, side, probable_pitcher_id(game, side))
    if player:
        return pitcher_quality_from_player(player)
    return {
        'STARTER_ERA': 4.25,
        'STARTER_WHIP': 1.30,
        'STARTER_K_BB': 2.35,
        'STARTER_IP_PER_START': 5.2,
    }


def lineup_context(box_score, side):
    if not box_score:
        return {'LINEUP_OPS': 0.710, 'LINEUP_OBP': 0.315, 'LINEUP_SLG': 0.395}
    team = box_team(box_score, side)
    players = team.get('players', {}) or {}
    batter_ids = [f'ID{player_id}' for player_id in team.get('batters', [])]
    ops_values = []
    obp_values = []
    slg_values = []
    for player_id in batter_ids[:9]:
        batting, _ = player_stat_profile(players.get(player_id) or {})
        ops_values.append(parse_number(batting.get('ops'), 0.710))
        obp_values.append(parse_number(batting.get('obp'), 0.315))
        slg_values.append(parse_number(batting.get('slg'), 0.395))
    if not ops_values:
        return {'LINEUP_OPS': 0.710, 'LINEUP_OBP': 0.315, 'LINEUP_SLG': 0.395}
    return {
        'LINEUP_OPS': sum(ops_values) / len(ops_values),
        'LINEUP_OBP': sum(obp_values) / len(obp_values),
        'LINEUP_SLG': sum(slg_values) / len(slg_values),
    }


def team_context_features(profile, game, box_score, side):
    context = {}
    context.update(starter_context(game, box_score, side))
    context.update(lineup_context(box_score, side))
    context['BULLPEN_ERA_7'] = parse_number(profile.get('bullpen_era_last_7'), 4.25)
    context['BULLPEN_WHIP_7'] = parse_number(profile.get('bullpen_whip_last_7'), 1.30)
    context['BULLPEN_FATIGUE_3'] = parse_number(profile.get('bullpen_fatigue_last_7'), 0.0)
    return context


def matchup_profiles(game, season=None, cutoff_date=None):
    profiles_by_team = None
    if cutoff_date:
        profiles_by_team = team_profiles_before_date(int(season or selected_mlb_season()), cutoff_date)
    home_profile = team_profile_for_game(game, 'home', season, profiles_by_team)
    away_profile = team_profile_for_game(game, 'away', season, profiles_by_team)
    if home_profile is None or away_profile is None:
        return None, None
    return home_profile, away_profile


def team_profile_from_box(box_score, side):
    team = box_team_info(box_score, side)
    if not team.get('id'):
        return None
    season = int_or_zero(team.get('season')) or selected_mlb_season()
    return team_metric_profile(team.get('id'), season, team_record_pct(box_score, side))


def play_progress(play):
    inning = int_or_zero(get_path(play, 'about', 'inning', default=0))
    if inning <= 0:
        return 0.0, 0, 0
    outs = int_or_zero(get_path(play, 'count', 'outs', default=0))
    is_top = bool(get_path(play, 'about', 'is_top_inning', default=True))
    completed_outs = (inning - 1) * 6 + outs
    if not is_top:
        completed_outs += 3
    return max(0.0, min(completed_outs / 54, 1.0)), inning, min(outs, 3)


def live_state_row(pregame_row, state, progress, inning, outs):
    row = {feature: pregame_row.get(feature, 0) for feature in PREGAME_FEATURES}
    row.update({
        'GAME_DATE': pregame_row.get('GAME_DATE', ''),
        'PREGAME_HOME_PROB': 0.5,
        'PROGRESS': progress,
        'INNING': inning,
        'OUTS': outs,
        'SCORE_DIFF': state['home']['runs'] - state['away']['runs'],
        'HIT_DIFF': state['home']['hits'] - state['away']['hits'],
        'TOTAL_BASE_DIFF': state['home']['total_bases'] - state['away']['total_bases'],
        'HOME_RUN_DIFF': state['home']['home_runs'] - state['away']['home_runs'],
        'WALK_DIFF': state['home']['walks'] - state['away']['walks'],
        'BATTING_K_EDGE': state['away']['batting_strikeouts'] - state['home']['batting_strikeouts'],
        'PITCHING_K_DIFF': state['home']['pitching_strikeouts'] - state['away']['pitching_strikeouts'],
        'ERROR_EDGE': state['away']['errors'] - state['home']['errors'],
        'LOB_EDGE': 0,
    })
    return row


def apply_play_to_live_state(state, play):
    half = (get_path(play, 'about', 'half_inning', default='') or '').lower()
    batting_side = 'away' if half == 'top' else 'home'
    fielding_side = 'home' if batting_side == 'away' else 'away'
    result = get_path(play, 'result', default={}) or {}
    event_type = str(result.get('event_type') or result.get('eventType') or '').lower()

    state['away']['runs'] = parse_number(result.get('away_score'), state['away']['runs'])
    state['home']['runs'] = parse_number(result.get('home_score'), state['home']['runs'])

    total_bases_by_event = {
        'single': 1,
        'double': 2,
        'triple': 3,
        'home_run': 4,
    }
    if event_type in total_bases_by_event:
        state[batting_side]['hits'] += 1
        state[batting_side]['total_bases'] += total_bases_by_event[event_type]
    if event_type == 'home_run':
        state[batting_side]['home_runs'] += 1
    if event_type in {'walk', 'intent_walk'}:
        state[batting_side]['walks'] += 1
    if 'strikeout' in event_type:
        state[batting_side]['batting_strikeouts'] += 1
        state[fielding_side]['pitching_strikeouts'] += 1
    if 'error' in event_type:
        state[fielding_side]['errors'] += 1


def real_live_feature_rows(game, pregame_row):
    game_id = get_path(game, 'game_pk')
    if not game_id:
        return []
    try:
        play_by_play = raw_game_play_by_play(int(game_id))
    except Exception:
        return []

    plays = play_by_play.get('all_plays') or play_by_play.get('allPlays') or []
    state = {
        'away': defaultdict(float),
        'home': defaultdict(float),
    }
    rows = []
    checkpoint_index = 0

    for play in sorted(plays, key=lambda item: int_or_zero(get_path(item, 'about', 'at_bat_index', default=0))):
        apply_play_to_live_state(state, play)
        progress, inning, outs = play_progress(play)
        while checkpoint_index < len(LIVE_PROGRESS_CHECKPOINTS) and progress >= LIVE_PROGRESS_CHECKPOINTS[checkpoint_index]:
            rows.append(live_state_row(pregame_row, state, progress, inning, outs))
            checkpoint_index += 1
    return rows


def build_training_examples(seasons):
    rows = []
    live_rows = []
    targets = []
    live_targets = []
    live_games_used = 0

    for season in seasons:
        totals_by_team = defaultdict(empty_team_totals)
        history_by_team = defaultdict(list)
        for game in sorted_regular_games(season):
            home_id = game_team_id(game, 'home')
            away_id = game_team_id(game, 'away')
            home_score = game_score(game, 'home')
            away_score = game_score(game, 'away')
            if home_score == away_score:
                update_team_totals_from_game(totals_by_team, game, history_by_team)
                continue

            if not home_id or not away_id:
                continue
            home_profile = profile_with_rolling_windows(totals_by_team[home_id], history_by_team[home_id])
            away_profile = profile_with_rolling_windows(totals_by_team[away_id], history_by_team[away_id])
            box_score = None
            if MLB_USE_HISTORICAL_BOX:
                try:
                    box_score = raw_game_box_score(int(get_path(game, 'game_pk')))
                except Exception:
                    box_score = None
            row = pregame_feature_row(
                home_profile,
                away_profile,
                team_context_features(home_profile, game, box_score, 'home'),
                team_context_features(away_profile, game, box_score, 'away'),
                game_context_features(game, 0.5),
            )
            row['GAME_DATE'] = get_path(game, 'official_date', default=get_path(game, 'game_date', default=''))
            target = int(home_score > away_score)
            rows.append(row)
            targets.append(target)

            if live_games_used < MLB_LIVE_TRAINING_GAMES:
                game_live_rows = real_live_feature_rows(game, row)
                if game_live_rows:
                    live_games_used += 1
                    for live_row in game_live_rows:
                        live_rows.append(live_row)
                        live_targets.append(target)
            update_team_totals_from_game(totals_by_team, game, history_by_team)

    if not rows:
        raise RuntimeError('No completed MLB games were available for sklearn training.')

    pregame_frame = pd.DataFrame(rows)
    y = pd.Series(targets, name='HOME_WIN')
    live_frame = pd.DataFrame(live_rows)
    live_y = pd.Series(live_targets, name='HOME_WIN')
    return pregame_frame, y, live_frame, live_y


def fit_pregame_model(X, y, c_value=0.8):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=c_value, class_weight='balanced'),
    )
    model.fit(X[PREGAME_FEATURES], y)
    return model


def fit_live_model(X, y):
    model = GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.045,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    )
    model.fit(X[LIVE_FEATURES], y)
    return model


def validation_metrics(model, X, y, features):
    probability = model.predict_proba(X[features])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        'accuracy': round(accuracy_score(y, prediction), 4),
        'log_loss': round(log_loss(y, probability, labels=[0, 1]), 4),
        'brier': round(brier_score_loss(y, probability), 4),
        'calibration': calibration_curve_bins(y, probability),
        'games': int(len(y)),
    }


def calibration_curve_bins(y_true, probabilities, bins=5):
    frame = pd.DataFrame({'actual': y_true, 'probability': probabilities})
    frame['bin'] = pd.cut(frame['probability'], bins=bins, labels=False, include_lowest=True)
    rows = []
    for _, group in frame.groupby('bin', observed=True):
        rows.append({
            'avg_probability': round(float(group['probability'].mean()), 4),
            'actual_rate': round(float(group['actual'].mean()), 4),
            'games': int(len(group)),
        })
    return rows


def aggregate_metric_rows(rows):
    games = sum(row.get('games', 0) for row in rows)
    if games <= 0:
        return {}
    return {
        'accuracy': round(sum(row['accuracy'] * row['games'] for row in rows) / games, 4),
        'log_loss': round(sum(row['log_loss'] * row['games'] for row in rows) / games, 4),
        'brier': round(sum(row['brier'] * row['games'] for row in rows) / games, 4),
        'games': int(games),
        'folds': len(rows),
        'fold_metrics': rows,
    }


def walk_forward_validation(X, y, features, fit_model):
    rows = []
    min_train = max(200, int(len(X) * 0.45))
    fold_size = max(50, int(len(X) * 0.12))
    start = min_train
    while start < len(X):
        end = min(start + fold_size, len(X))
        if y.iloc[:start].nunique() == 2 and end > start:
            model = fit_model(X.iloc[:start], y.iloc[:start])
            rows.append(validation_metrics(model, X.iloc[start:end], y.iloc[start:end], features))
        start = end
    return aggregate_metric_rows(rows)


def tune_pregame_regularization(X, y):
    candidates = (0.25, 0.5, 0.8, 1.2)
    results = {}
    best_c = 0.8
    best_score = None
    for c_value in candidates:
        metrics = walk_forward_validation(
            X,
            y,
            PREGAME_FEATURES,
            lambda train_x, train_y, value=c_value: fit_pregame_model(train_x, train_y, value),
        )
        if not metrics:
            continue
        results[str(c_value)] = metrics
        score = (metrics['log_loss'], metrics['brier'])
        if best_score is None or score < best_score:
            best_score = score
            best_c = c_value
    return best_c, results


def train_mlb_models(seasons):
    print(f"--- Training MLB sklearn models on seasons: {seasons} ---")
    pregame_frame, y, live_frame, live_y = build_training_examples(seasons)
    pregame_frame['HOME_WIN'] = y
    pregame_frame = pregame_frame.sort_values('GAME_DATE').reset_index(drop=True)
    y = pregame_frame.pop('HOME_WIN').reset_index(drop=True)
    validation = {}
    selected_c, c_results = tune_pregame_regularization(pregame_frame, y)
    validation['pregame'] = {
        'selected_c': selected_c,
        'walk_forward': c_results.get(str(selected_c), {}),
        'candidates': c_results,
    }

    pregame_model = fit_pregame_model(pregame_frame, y, selected_c)
    live_model = None
    if not live_frame.empty:
        live_frame['HOME_WIN'] = live_y
        live_frame = live_frame.sort_values('GAME_DATE').reset_index(drop=True)
        live_y = live_frame.pop('HOME_WIN').reset_index(drop=True)
        live_split_at = max(1, int(len(live_frame) * 0.8))
        if live_split_at < len(live_frame) and live_y.iloc[:live_split_at].nunique() == 2:
            live_train = live_frame.iloc[:live_split_at].copy()
            live_test = live_frame.iloc[live_split_at:].copy()
            live_pregame_model = fit_pregame_model(live_train, live_y.iloc[:live_split_at])
            live_train['PREGAME_HOME_PROB'] = live_pregame_model.predict_proba(live_train[PREGAME_FEATURES])[:, 1]
            live_test['PREGAME_HOME_PROB'] = live_pregame_model.predict_proba(live_test[PREGAME_FEATURES])[:, 1]
            live_validation_model = fit_live_model(live_train, live_y.iloc[:live_split_at])
            validation['live'] = validation_metrics(
                live_validation_model,
                live_test,
                live_y.iloc[live_split_at:],
                LIVE_FEATURES,
            )
        live_frame['PREGAME_HOME_PROB'] = pregame_model.predict_proba(live_frame[PREGAME_FEATURES])[:, 1]
        live_model = fit_live_model(live_frame, live_y)
    else:
        validation['live'] = {
            'status': 'unavailable',
            'reason': 'No historical MLB play-by-play snapshots were available; synthetic live training data is disabled.',
        }

    payload = {
        'version': MODEL_CACHE_VERSION,
        'seasons': seasons,
        'pregame_model': pregame_model,
        'live_model': live_model,
        'pregame_features': PREGAME_FEATURES,
        'live_features': LIVE_FEATURES,
        'metrics': validation,
    }
    save_model_cache(payload, seasons)
    print(f"--- MLB sklearn training complete: {validation} ---")
    return payload


def ensure_mlb_model():
    global _MODEL_CACHE, _METRICS_CACHE
    seasons = mlb_training_seasons()
    with _MODEL_LOCK:
        if _MODEL_CACHE is not None or load_cached_model(seasons):
            return _MODEL_CACHE
        _MODEL_CACHE = train_mlb_models(seasons)
        _METRICS_CACHE = _MODEL_CACHE.get('metrics')
        return _MODEL_CACHE


def is_mlb_model_ready():
    seasons = mlb_training_seasons()
    return _MODEL_CACHE is not None or model_cache_path(seasons).exists()


def start_mlb_training():
    global _TRAINING
    seasons = mlb_training_seasons()
    with _MODEL_LOCK:
        if _MODEL_CACHE is not None or load_cached_model(seasons):
            return 'ready'
        if _TRAINING:
            return 'training'
        _TRAINING = True

    def train_in_background():
        global _MODEL_CACHE, _METRICS_CACHE, _TRAINING
        try:
            payload = train_mlb_models(seasons)
            with _MODEL_LOCK:
                _MODEL_CACHE = payload
                _METRICS_CACHE = payload.get('metrics')
        except Exception as error:
            print(f"--- MLB sklearn training failed: {error} ---")
        finally:
            with _MODEL_LOCK:
                _TRAINING = False

    threading.Thread(target=train_in_background, name='mlb-sklearn-training', daemon=True).start()
    return 'training'


def mlb_model_diagnostics():
    if _METRICS_CACHE is not None:
        return _METRICS_CACHE
    seasons = mlb_training_seasons()
    if load_cached_model(seasons):
        return _METRICS_CACHE
    return None


def model_pregame_probability(home_profile, away_profile, home_context=None, away_context=None, game_context=None):
    payload = ensure_mlb_model()
    row = pd.DataFrame([pregame_feature_row(home_profile, away_profile, home_context, away_context, game_context)])
    return float(payload['pregame_model'].predict_proba(row[PREGAME_FEATURES])[0][1])


def metric_row(label, home_value, away_value, higher_is_better=True, precision=3):
    home = parse_number(home_value)
    away = parse_number(away_value)
    edge = home - away if higher_is_better else away - home
    leader = 'home' if edge >= 0 else 'away'
    return {
        'label': label,
        'home': round(home, precision),
        'away': round(away, precision),
        'edge': round(edge, precision),
        'leader': leader,
    }


def prediction_metrics(home_profile, away_profile, home_context=None, away_context=None, game_context=None):
    home_context = home_context or {}
    away_context = away_context or {}
    game_context = game_context or {}
    rows = [
        metric_row('Runs per game', home_profile['runs_per_game'], away_profile['runs_per_game'], True, 2),
        metric_row('OPS', home_profile['ops'], away_profile['ops'], True, 3),
        metric_row('OBP', home_profile['obp'], away_profile['obp'], True, 3),
        metric_row('SLG', home_profile['slg'], away_profile['slg'], True, 3),
        metric_row('Last 7 RPG', home_profile.get('runs_per_game_last_7'), away_profile.get('runs_per_game_last_7'), True, 2),
        metric_row('Last 14 OPS', home_profile.get('ops_last_14'), away_profile.get('ops_last_14'), True, 3),
        metric_row('Last 30 record', home_profile.get('record_pct_last_30'), away_profile.get('record_pct_last_30'), True, 3),
        metric_row('Starter ERA', home_context.get('STARTER_ERA'), away_context.get('STARTER_ERA'), False, 2),
        metric_row('Starter WHIP', home_context.get('STARTER_WHIP'), away_context.get('STARTER_WHIP'), False, 2),
        metric_row('Starter K/BB', home_context.get('STARTER_K_BB'), away_context.get('STARTER_K_BB'), True, 2),
        metric_row('Lineup OPS', home_context.get('LINEUP_OPS'), away_context.get('LINEUP_OPS'), True, 3),
        metric_row('Bullpen ERA 7', home_context.get('BULLPEN_ERA_7'), away_context.get('BULLPEN_ERA_7'), False, 2),
        metric_row('Bullpen fatigue', home_context.get('BULLPEN_FATIGUE_3'), away_context.get('BULLPEN_FATIGUE_3'), False, 0),
        metric_row('ERA', home_profile['era'], away_profile['era'], False, 2),
        metric_row('WHIP', home_profile['whip'], away_profile['whip'], False, 2),
        metric_row('Runs allowed per game', home_profile['runs_allowed_per_game'], away_profile['runs_allowed_per_game'], False, 2),
        metric_row('Pitching K/BB', home_profile['strikeout_walk_ratio'], away_profile['strikeout_walk_ratio'], True, 2),
        metric_row('Win pct', home_profile['record_pct'], away_profile['record_pct'], True, 3),
    ]
    rows.extend([
        {
            'label': 'Park factor',
            'home': round(parse_number(game_context.get('PARK_FACTOR'), DEFAULT_PARK_FACTOR), 3),
            'away': DEFAULT_PARK_FACTOR,
            'edge': round(parse_number(game_context.get('PARK_FACTOR'), DEFAULT_PARK_FACTOR) - DEFAULT_PARK_FACTOR, 3),
            'leader': 'home',
        },
        {
            'label': 'Weather temp',
            'home': round(parse_number(game_context.get('TEMP_F'), 72.0), 0),
            'away': 72,
            'edge': round(parse_number(game_context.get('TEMP_F'), 72.0) - 72, 0),
            'leader': 'home',
        },
        {
            'label': 'Market home implied',
            'home': round(parse_number(game_context.get('MARKET_HOME_IMPLIED'), 0.5), 3),
            'away': round(1 - parse_number(game_context.get('MARKET_HOME_IMPLIED'), 0.5), 3),
            'edge': round(parse_number(game_context.get('MARKET_HOME_IMPLIED'), 0.5) - 0.5, 3),
            'leader': 'home' if parse_number(game_context.get('MARKET_HOME_IMPLIED'), 0.5) >= 0.5 else 'away',
        },
    ])
    return rows


def live_metric_rows(home_live, away_live):
    return [
        metric_row('Runs', home_live['runs'], away_live['runs'], True, 0),
        metric_row('Hits', home_live['hits'], away_live['hits'], True, 0),
        metric_row('Total bases', home_live['total_bases'], away_live['total_bases'], True, 0),
        metric_row('Home runs', home_live['home_runs'], away_live['home_runs'], True, 0),
        metric_row('Walks', home_live['walks'], away_live['walks'], True, 0),
        metric_row('Batting strikeouts', home_live['batting_strikeouts'], away_live['batting_strikeouts'], False, 0),
        metric_row('Pitching strikeouts', home_live['pitching_strikeouts'], away_live['pitching_strikeouts'], True, 0),
        metric_row('Errors', home_live['errors'], away_live['errors'], False, 0),
        metric_row('Left on base', home_live['left_on_base'], away_live['left_on_base'], False, 0),
    ]


def mlb_predict_game(game_id, selected_date=None):
    selected_date = selected_date or datetime.date.today().isoformat()
    game = find_mlb_schedule_game(game_id, selected_date)
    if not game:
        raise RuntimeError('MLB game was not found on the selected scoreboard date.')

    home_team = game_team(game, 'home')
    away_team = game_team(game, 'away')
    season = mlb_season_for_date(selected_date)
    home_profile, away_profile = matchup_profiles(game, season, selected_date)
    if home_profile is None or away_profile is None:
        raise RuntimeError('MLB team profiles are unavailable for this game.')
    try:
        box_score = raw_game_box_score(int(game_id))
    except Exception:
        box_score = None
    market_probability = market_home_implied_probability(game)
    home_context = team_context_features(home_profile, game, box_score, 'home')
    away_context = team_context_features(away_profile, game, box_score, 'away')
    context = game_context_features(game, market_probability)
    model_home_probability = model_pregame_probability(home_profile, away_profile, home_context, away_context, context)
    if MLB_USE_MARKET_FEATURE and market_probability != 0.5:
        home_probability = sigmoid((logit(model_home_probability) * 0.75) + (logit(market_probability) * 0.25))
    else:
        home_probability = model_home_probability
    winner = home_team.get('name') if home_probability >= 0.5 else away_team.get('name')

    return {
        'winner': winner,
        'confidence': max(home_probability, 1 - home_probability) * 100,
        'probabilities': {'home': home_probability, 'away': 1 - home_probability},
        'modelProbabilities': {'home': model_home_probability, 'away': 1 - model_home_probability},
        'metrics': prediction_metrics(home_profile, away_profile, home_context, away_context, context),
        'analysis': (
            f"Sklearn MLB model trained on {', '.join(str(season) for season in mlb_training_seasons())} using chronological, date-capped team profiles with starter, lineup, bullpen, "
            "park/weather, and last-7/14/30 form features. Validation uses walk-forward folds scored by log loss "
            f"and Brier score. Market odds blending is {'enabled' if MLB_USE_MARKET_FEATURE else 'disabled'}."
        ),
    }


def game_progress(line_score):
    inning = int_or_zero(line_score.get('current_inning'))
    outs = int_or_zero(line_score.get('outs'))
    is_top = bool(line_score.get('is_top_inning'))
    if inning <= 0:
        return 0.0, inning
    completed_outs = (inning - 1) * 6 + outs
    if not is_top:
        completed_outs += 3
    return max(0.0, min(completed_outs / 54, 1.0)), inning


def live_team_profile(line_score, box_score, side):
    team_stats = get_path(box_score, 'teams', side, 'team_stats', default={}) or {}
    batting = team_stats.get('batting', {})
    pitching = team_stats.get('pitching', {})
    line_team = get_path(line_score, 'teams', side, default={}) or {}
    return {
        'runs': parse_number(line_team.get('runs') if line_team.get('runs') is not None else batting.get('runs')),
        'hits': parse_number(line_team.get('hits') if line_team.get('hits') is not None else batting.get('hits')),
        'errors': parse_number(line_team.get('errors'), parse_number(get_path(team_stats, 'fielding', 'errors'))),
        'left_on_base': parse_number(line_team.get('left_on_base') if line_team.get('left_on_base') is not None else batting.get('leftOnBase')),
        'total_bases': parse_number(batting.get('totalBases')),
        'home_runs': parse_number(batting.get('homeRuns')),
        'walks': parse_number(batting.get('baseOnBalls')),
        'batting_strikeouts': parse_number(batting.get('strikeOuts')),
        'pitching_strikeouts': parse_number(pitching.get('strikeOuts')),
        'pitching_walks': parse_number(pitching.get('baseOnBalls')),
        'pitches': parse_number(pitching.get('numberOfPitches') or pitching.get('pitchesThrown')),
        'strike_percentage': parse_number(pitching.get('strikePercentage')),
    }


def live_feature_row(prior_probability, line_score, home_live, away_live, progress, inning):
    return {
        'PREGAME_HOME_PROB': prior_probability,
        'PROGRESS': progress,
        'INNING': inning,
        'OUTS': int_or_zero(line_score.get('outs')),
        'SCORE_DIFF': home_live['runs'] - away_live['runs'],
        'HIT_DIFF': home_live['hits'] - away_live['hits'],
        'TOTAL_BASE_DIFF': home_live['total_bases'] - away_live['total_bases'],
        'HOME_RUN_DIFF': home_live['home_runs'] - away_live['home_runs'],
        'WALK_DIFF': home_live['walks'] - away_live['walks'],
        'BATTING_K_EDGE': away_live['batting_strikeouts'] - home_live['batting_strikeouts'],
        'PITCHING_K_DIFF': home_live['pitching_strikeouts'] - away_live['pitching_strikeouts'],
        'ERROR_EDGE': away_live['errors'] - home_live['errors'],
        'LOB_EDGE': away_live['left_on_base'] - home_live['left_on_base'],
    }


def clamp_probability(value):
    return max(0.001, min(parse_number(value, 0.5), 0.999))


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def logit(value):
    value = clamp_probability(value)
    return math.log(value / (1 - value))


def heuristic_live_probability(prior_probability, home_live, away_live, progress):
    score_diff = home_live['runs'] - away_live['runs']
    if progress >= 1 and score_diff != 0:
        return 1.0 if score_diff > 0 else 0.0

    hit_diff = home_live['hits'] - away_live['hits']
    base_diff = home_live['total_bases'] - away_live['total_bases']
    # Early MLB leads are fragile. Let scoreboard leverage grow slowly early,
    # then accelerate in later innings as remaining outs disappear.
    score_weight = 0.38 + (1.85 * (progress ** 2.4))
    prior_weight = max(0.25, 1 - (progress * 0.85))
    contact_signal = max(-0.35, min(0.35, (hit_diff * 0.025) + (base_diff * 0.015)))
    state_signal = (
        score_diff * score_weight
        + contact_signal
    )
    return sigmoid((logit(prior_probability) * prior_weight) + state_signal)


def season_prior_from_box(box_score):
    home_profile = team_profile_from_box(box_score, 'home')
    away_profile = team_profile_from_box(box_score, 'away')
    if home_profile is None or away_profile is None:
        return 0.5
    return model_pregame_probability(home_profile, away_profile)


def live_home_probability(line_score, box_score):
    home_live = live_team_profile(line_score, box_score, 'home')
    away_live = live_team_profile(line_score, box_score, 'away')
    progress, inning = game_progress(line_score)
    prior_probability = season_prior_from_box(box_score)
    payload = ensure_mlb_model()
    row = pd.DataFrame([live_feature_row(prior_probability, line_score, home_live, away_live, progress, inning)])
    live_model = payload.get('live_model')
    if live_model is None:
        home_probability = heuristic_live_probability(prior_probability, home_live, away_live, progress)
    else:
        home_probability = float(live_model.predict_proba(row[LIVE_FEATURES])[0][1])
    return home_probability, home_live, away_live, progress


def mlb_live_projection(game_id):
    line_score = dump_model(mlb_client().get_game_line_score(int(game_id)))
    box_score = dump_model(mlb_client().get_game_box_score(int(game_id)))
    if int_or_zero(line_score.get('current_inning')) <= 0:
        raise RuntimeError('Game has not started yet')
    home_team = box_team_info(box_score, 'home')
    away_team = box_team_info(box_score, 'away')
    home_name = home_team.get('name') or 'Home'
    away_name = away_team.get('name') or 'Away'
    home_probability, home_live, away_live, progress = live_home_probability(line_score, box_score)
    home_score = int_or_zero(home_live['runs'])
    away_score = int_or_zero(away_live['runs'])
    inning = int_or_zero(line_score.get('current_inning'))
    clock = (
        f"{line_score.get('inning_half') or line_score.get('inning_state') or ''} "
        f"{line_score.get('current_inning_ordinal') or ordinal_inning(inning)}"
    ).strip()
    is_final = progress >= 1 and home_score != away_score
    winner = home_name if home_probability >= 0.5 else away_name
    if is_final and home_score != away_score:
        winner = home_name if home_score > away_score else away_name
    payload = ensure_mlb_model()
    live_analysis = (
        f"Sklearn live MLB model trained from real play-by-play snapshots for {', '.join(str(season) for season in mlb_training_seasons())} using score, hits, total bases, home runs, walks, strikeouts, errors, inning progress, and pregame prior."
        if payload.get('live_model') is not None
        else 'Live MLB projection uses the corrected pregame prior plus current score/base-hit state because historical play-by-play snapshots were unavailable during training; synthetic live training data is disabled.'
    )

    return {
        'winner': winner,
        'confidence': round(max(home_probability, 1 - home_probability) * 100, 2),
        'probabilities': {'home': home_probability, 'away': 1 - home_probability},
        'modelProbabilities': {'home': home_probability, 'away': 1 - home_probability},
        'home': short_team_name(home_name),
        'away': short_team_name(away_name),
        'homeFullName': home_name,
        'awayFullName': away_name,
        'score': f'{away_score} - {home_score}',
        'clock': 'FINAL' if is_final else clock,
        'period': inning,
        'isFinal': is_final,
        'metrics': live_metric_rows(home_live, away_live),
        'analysis': live_analysis,
    }
