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
PREGAME_FEATURES = [f'DIFF_{feature}' for feature in PROFILE_FEATURES] + ['HOME_RECORD_PCT', 'AWAY_RECORD_PCT']
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
MODEL_CACHE_VERSION = 5
MODEL_CACHE_DIR = Path(os.environ.get('MLB_MODEL_CACHE_DIR', Path(__file__).with_name('model_cache')))
MLB_TRAINING_SEASONS = max(2, int(os.environ.get('MLB_TRAINING_SEASONS', '2')))
MLB_USE_HISTORICAL_BOX = os.environ.get('MLB_USE_HISTORICAL_BOX', '0') == '1'
MLB_LIVE_TRAINING_GAMES = max(0, int(os.environ.get('MLB_LIVE_TRAINING_GAMES', '0')))
LIVE_PROGRESS_CHECKPOINTS = (0.2, 0.45, 0.7, 0.9)

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
    }


def add_totals(left, right):
    for key, value in right.items():
        left[key] += value
    return left


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


def past_completed_mlb_seasons(count=MLB_TRAINING_SEASONS):
    current = selected_mlb_season()
    return list(range(current - count, current))


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


def pregame_feature_row(home_profile, away_profile):
    row = {}
    for feature in PROFILE_FEATURES:
        row[f'DIFF_{feature}'] = parse_number(home_profile.get(feature)) - parse_number(away_profile.get(feature))
    row['HOME_RECORD_PCT'] = parse_number(home_profile.get('record_pct'), 0.5)
    row['AWAY_RECORD_PCT'] = parse_number(away_profile.get('record_pct'), 0.5)
    return row


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


def update_team_totals_from_game(totals_by_team, game):
    game_id = get_path(game, 'game_pk')
    home_id = game_team_id(game, 'home')
    away_id = game_team_id(game, 'away')
    if not home_id or not away_id:
        return

    home_score = game_score(game, 'home')
    away_score = game_score(game, 'away')
    if home_score == away_score:
        return

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


@lru_cache(maxsize=256)
def team_profiles_before_date(season, cutoff_date):
    totals_by_team = defaultdict(empty_team_totals)
    for game in sorted_regular_games(int(season)):
        game_date = get_path(game, 'official_date', default='') or ''
        if game_date >= cutoff_date:
            break
        update_team_totals_from_game(totals_by_team, game)
    return {
        int(team_id): team_profile_from_totals(totals)
        for team_id, totals in totals_by_team.items()
    }


def team_profile_for_game(game, side, season=None, profiles_by_team=None):
    team = game_team(game, side)
    if not team.get('id'):
        return None
    if profiles_by_team is not None:
        return profiles_by_team.get(int(team.get('id')), team_profile_from_totals(empty_team_totals()))
    return team_metric_profile(
        team.get('id'),
        season or selected_mlb_season(),
        team_record_pct(game, side),
    )


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
        for game in sorted_regular_games(season):
            home_id = game_team_id(game, 'home')
            away_id = game_team_id(game, 'away')
            home_score = game_score(game, 'home')
            away_score = game_score(game, 'away')
            if home_score == away_score:
                update_team_totals_from_game(totals_by_team, game)
                continue

            if not home_id or not away_id:
                continue
            home_profile = team_profile_from_totals(totals_by_team[home_id])
            away_profile = team_profile_from_totals(totals_by_team[away_id])
            row = pregame_feature_row(home_profile, away_profile)
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
            update_team_totals_from_game(totals_by_team, game)

    if not rows:
        raise RuntimeError('No completed MLB games were available for sklearn training.')

    pregame_frame = pd.DataFrame(rows)
    y = pd.Series(targets, name='HOME_WIN')
    live_frame = pd.DataFrame(live_rows)
    live_y = pd.Series(live_targets, name='HOME_WIN')
    return pregame_frame, y, live_frame, live_y


def fit_pregame_model(X, y):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.8, class_weight='balanced'),
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
        'games': int(len(y)),
    }


def train_mlb_models(seasons):
    print(f"--- Training MLB sklearn models on seasons: {seasons} ---")
    pregame_frame, y, live_frame, live_y = build_training_examples(seasons)
    pregame_frame['HOME_WIN'] = y
    pregame_frame = pregame_frame.sort_values('GAME_DATE').reset_index(drop=True)
    y = pregame_frame.pop('HOME_WIN').reset_index(drop=True)
    split_at = max(1, int(len(pregame_frame) * 0.8))

    validation = {}
    if split_at < len(pregame_frame) and y.iloc[:split_at].nunique() == 2:
        validation_model = fit_pregame_model(pregame_frame.iloc[:split_at], y.iloc[:split_at])
        validation['pregame'] = validation_metrics(
            validation_model,
            pregame_frame.iloc[split_at:],
            y.iloc[split_at:],
            PREGAME_FEATURES,
        )

    pregame_model = fit_pregame_model(pregame_frame, y)
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
    seasons = past_completed_mlb_seasons()
    with _MODEL_LOCK:
        if _MODEL_CACHE is not None or load_cached_model(seasons):
            return _MODEL_CACHE
        _MODEL_CACHE = train_mlb_models(seasons)
        _METRICS_CACHE = _MODEL_CACHE.get('metrics')
        return _MODEL_CACHE


def is_mlb_model_ready():
    seasons = past_completed_mlb_seasons()
    return _MODEL_CACHE is not None or model_cache_path(seasons).exists()


def start_mlb_training():
    global _TRAINING
    seasons = past_completed_mlb_seasons()
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
    seasons = past_completed_mlb_seasons()
    if load_cached_model(seasons):
        return _METRICS_CACHE
    return None


def model_pregame_probability(home_profile, away_profile):
    payload = ensure_mlb_model()
    row = pd.DataFrame([pregame_feature_row(home_profile, away_profile)])
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


def prediction_metrics(home_profile, away_profile):
    return [
        metric_row('Runs per game', home_profile['runs_per_game'], away_profile['runs_per_game'], True, 2),
        metric_row('OPS', home_profile['ops'], away_profile['ops'], True, 3),
        metric_row('OBP', home_profile['obp'], away_profile['obp'], True, 3),
        metric_row('SLG', home_profile['slg'], away_profile['slg'], True, 3),
        metric_row('ERA', home_profile['era'], away_profile['era'], False, 2),
        metric_row('WHIP', home_profile['whip'], away_profile['whip'], False, 2),
        metric_row('Runs allowed per game', home_profile['runs_allowed_per_game'], away_profile['runs_allowed_per_game'], False, 2),
        metric_row('Pitching K/BB', home_profile['strikeout_walk_ratio'], away_profile['strikeout_walk_ratio'], True, 2),
        metric_row('Win pct', home_profile['record_pct'], away_profile['record_pct'], True, 3),
    ]


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
    home_probability = model_pregame_probability(home_profile, away_profile)
    winner = home_team.get('name') if home_probability >= 0.5 else away_team.get('name')

    return {
        'winner': winner,
        'confidence': max(home_probability, 1 - home_probability) * 100,
        'probabilities': {'home': home_probability, 'away': 1 - home_probability},
        'modelProbabilities': {'home': home_probability, 'away': 1 - home_probability},
        'metrics': prediction_metrics(home_profile, away_profile),
        'analysis': f"Sklearn MLB model trained on chronological, date-capped rolling team profiles from {', '.join(str(season) for season in past_completed_mlb_seasons())}.",
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
        f"Sklearn live MLB model trained from real play-by-play snapshots for {', '.join(str(season) for season in past_completed_mlb_seasons())} using score, hits, total bases, home runs, walks, strikeouts, errors, inning progress, and pregame prior."
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
