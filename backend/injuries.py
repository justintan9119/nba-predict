import json
import math
import os
import re
from pathlib import Path

INJURY_POINT_LOGIT_WEIGHT = 0.08
DEFAULT_INJURY_FILE = Path(__file__).with_name('injuries.json')


def _normalize_name(name):
    return re.sub(r'[^a-z0-9]+', '', str(name or '').lower())


def _load_injury_data():
    path = Path(os.environ.get('INJURY_REPORT_FILE', DEFAULT_INJURY_FILE))
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _matching_team_entry(league_report, team_name):
    normalized_team = _normalize_name(team_name)
    for key, value in league_report.items():
        normalized_key = _normalize_name(key)
        if normalized_key == normalized_team or normalized_key in normalized_team or normalized_team in normalized_key:
            return value
    return None


def _player_impact(player):
    if not isinstance(player, dict):
        return 0.0
    try:
        return max(0.0, float(player.get('impact', 0)))
    except (TypeError, ValueError):
        return 0.0


def _team_impact(entry):
    if entry is None:
        return 0.0
    if isinstance(entry, (int, float)):
        return max(0.0, float(entry))
    if isinstance(entry, list):
        return sum(_player_impact(player) for player in entry)
    if isinstance(entry, dict):
        explicit_impact = entry.get('impact')
        if explicit_impact is not None:
            try:
                return max(0.0, float(explicit_impact))
            except (TypeError, ValueError):
                return 0.0
        players = entry.get('players', [])
        if isinstance(players, list):
            return sum(_player_impact(player) for player in players)
    return 0.0


def team_injury_impact(team_name, league='nba'):
    data = _load_injury_data()
    league_report = data.get(league, {}) if isinstance(data, dict) else {}
    if not isinstance(league_report, dict):
        return 0.0
    return _team_impact(_matching_team_entry(league_report, team_name))


def injury_adjusted_home_probability(home_probability, home_team, away_team, league='nba'):
    home_probability = min(max(float(home_probability), 0.01), 0.99)
    home_impact = team_injury_impact(home_team, league)
    away_impact = team_injury_impact(away_team, league)
    impact_delta = away_impact - home_impact

    if impact_delta == 0:
        return home_probability

    logit = math.log(home_probability / (1 - home_probability))
    adjusted_logit = logit + (impact_delta * INJURY_POINT_LOGIT_WEIGHT)
    return 1 / (1 + math.exp(-adjusted_logit))