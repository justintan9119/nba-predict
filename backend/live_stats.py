import datetime
import re

import requests
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.library.parameters import LeagueIDNullable


LIVE_BOX_SCORE_URLS = {
    'nba': ('https://cdn.nba.com/static/json/liveData/boxscore', 'https://www.nba.com/'),
    'wnba': ('https://cdn.wnba.com/static/json/liveData/boxscore', 'https://www.wnba.com/'),
}
LEAGUE_IDS = {
    'nba': LeagueIDNullable.nba,
    'wnba': LeagueIDNullable.wnba,
}


def current_season(league):
    now = datetime.datetime.now()
    year = now.year
    if league == 'wnba':
        if now.month < 5:
            year -= 1
        return str(year)
    if now.month < 10:
        year -= 1
    return f'{year}-{(year + 1) % 100:02d}'


def number_average(value):
    return round(float(value or 0), 1)


def player_stats(player):
    stats = player.get('statistics', {})
    made = stats.get('fieldGoalsMade', 0)
    attempted = stats.get('fieldGoalsAttempted', 0)
    threes_made = stats.get('threePointersMade', 0)
    threes_attempted = stats.get('threePointersAttempted', 0)
    return {
        'name': player.get('name', 'Unknown Player'),
        'points': stats.get('points', 0),
        'rebounds': stats.get('reboundsTotal', 0),
        'assists': stats.get('assists', 0),
        'fieldGoals': f'{made}-{attempted}',
        'threePointers': f'{threes_made}-{threes_attempted}',
        'steals': stats.get('steals', 0),
        'blocks': stats.get('blocks', 0),
    }


def normalize_name(name):
    return re.sub(r'[^a-z0-9]+', '', str(name or '').lower())


def player_average_row(row):
    return {
        'name': row.get('PLAYER_NAME', 'Unknown Player'),
        'points': number_average(row.get('PTS')),
        'rebounds': number_average(row.get('REB')),
        'assists': number_average(row.get('AST')),
        'fieldGoals': f"{number_average(row.get('FGM'))}-{number_average(row.get('FGA'))}",
        'threePointers': f"{number_average(row.get('FG3M'))}-{number_average(row.get('FG3A'))}",
        'steals': number_average(row.get('STL')),
        'blocks': number_average(row.get('BLK')),
    }


def team_player_averages(league, team_id):
    if not team_id:
        return []
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            league_id_nullable=LEAGUE_IDS.get(league, LeagueIDNullable.nba),
            season=current_season(league),
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            team_id_nullable=str(team_id),
            timeout=10,
        )
        frame = stats.get_data_frames()[0]
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return []
    if frame.empty:
        return []
    frame = frame.sort_values(['MIN', 'PTS', 'REB', 'AST'], ascending=False).head(10)
    return [player_average_row(row) for row in frame.to_dict('records')]


def pregame_player_averages(league, away_team_id=None, home_team_id=None):
    return {
        'hasStarted': False,
        'away': team_player_averages(league, away_team_id),
        'home': team_player_averages(league, home_team_id),
    }


def parse_espn_players(team_data):
    statistics = team_data.get('statistics', [{}])[0]
    keys = statistics.get('keys', [])
    rows = statistics.get('athletes', [])
    key_index = {key: index for index, key in enumerate(keys)}

    def stat(row, key, default='0'):
        index = key_index.get(key)
        values = row.get('stats', [])
        return values[index] if index is not None and index < len(values) else default

    players = []
    for row in rows:
        if row.get('didNotPlay'):
            continue
        players.append({
            'name': row.get('athlete', {}).get('displayName', 'Unknown Player'),
            'points': stat(row, 'points'),
            'rebounds': stat(row, 'rebounds'),
            'assists': stat(row, 'assists'),
            'fieldGoals': stat(row, 'fieldGoalsMade-fieldGoalsAttempted', '-'),
            'threePointers': stat(row, 'threePointFieldGoalsMade-threePointFieldGoalsAttempted', '-'),
            'steals': stat(row, 'steals'),
            'blocks': stat(row, 'blocks'),
        })
    return players


def espn_live_game_stats(league, game_date, away_team, home_team):
    sport_league = 'wnba' if league == 'wnba' else 'nba'
    date = datetime.datetime.strptime(game_date, '%Y-%m-%d').strftime('%Y%m%d')
    scoreboard = requests.get(
        f'https://site.api.espn.com/apis/site/v2/sports/basketball/{sport_league}/scoreboard',
        params={'dates': date},
        timeout=10,
    ).json()
    away_key = normalize_name(away_team)
    home_key = normalize_name(home_team)
    event_id = None
    for event in scoreboard.get('events', []):
        names = [normalize_name(team['team']['displayName']) for team in event['competitions'][0]['competitors']]
        if any(away_key in name or name in away_key for name in names) and any(home_key in name or name in home_key for name in names):
            event_id = event['id']
            break
    if not event_id:
        raise ValueError('The matching ESPN game was not found.')

    summary = requests.get(
        f'https://site.api.espn.com/apis/site/v2/sports/basketball/{sport_league}/summary',
        params={'event': event_id},
        timeout=10,
    ).json()
    teams = summary.get('boxscore', {}).get('players', [])
    players_by_team = {
        normalize_name(team.get('team', {}).get('displayName')): parse_espn_players(team)
        for team in teams
    }
    return {
        'hasStarted': bool(players_by_team),
        'away': next((players for name, players in players_by_team.items() if away_key in name or name in away_key), []),
        'home': next((players for name, players in players_by_team.items() if home_key in name or name in home_key), []),
    }


def live_game_stats(game_id, league='nba', game_date=None, away_team=None, home_team=None, away_team_id=None, home_team_id=None):
    league = league if league in LIVE_BOX_SCORE_URLS else 'nba'
    base_url, referer = LIVE_BOX_SCORE_URLS[league]
    try:
        response = requests.get(
            f'{base_url}/boxscore_{game_id}.json',
            headers={'Referer': referer, 'User-Agent': 'Mozilla/5.0'},
            timeout=10,
        )
        response.raise_for_status()
        game = response.json()['game']
        if game.get('gameStatus') == 1:
            return pregame_player_averages(league, away_team_id, home_team_id)
    except (requests.RequestException, ValueError, KeyError):
        if game_date and away_team and home_team:
            stats = espn_live_game_stats(league, game_date, away_team, home_team)
            if stats.get('hasStarted') or stats.get('away') or stats.get('home'):
                return stats
        if away_team_id or home_team_id:
            return pregame_player_averages(league, away_team_id, home_team_id)
        raise

    def team_players(team):
        players = [player_stats(player) for player in team.get('players', []) if not player.get('notPlayingReason')]
        return sorted(players, key=lambda player: (player['points'], player['rebounds'], player['assists']), reverse=True)

    return {
        'hasStarted': True,
        'away': team_players(game['awayTeam']),
        'home': team_players(game['homeTeam']),
    }
