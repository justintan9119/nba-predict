from mlb_predict import TEAM_SIDES, box_team, dump_model, get_path, int_or_zero, mlb_client, parse_number


def has_stat_values(stats):
    return any(parse_number(value) for value in stats.values() if isinstance(value, (int, float, str)))


def batting_line(stats):
    hits = int_or_zero(stats.get('hits'))
    at_bats = int_or_zero(stats.get('atBats'))
    return f'{hits}-{at_bats}'


def pitching_outs(innings_pitched):
    if innings_pitched in (None, '', '-.--', '-'):
        return 0
    text = str(innings_pitched)
    whole, _, partial = text.partition('.')
    outs = int_or_zero(whole) * 3
    if partial:
        outs += min(int_or_zero(partial[0]), 2)
    return outs


def calculated_era(earned_runs, innings_pitched, api_era=None):
    if api_era not in (None, '', '-.--', '-'):
        return api_era
    outs = pitching_outs(innings_pitched)
    if outs <= 0:
        return '-.--'
    return f'{(parse_number(earned_runs) * 9 / (outs / 3)):.2f}'


def pitch_count_strikes(pitching):
    pitch_count = int_or_zero(pitching.get('numberOfPitches') or pitching.get('pitchesThrown'))
    strikes = int_or_zero(pitching.get('strikes'))
    return pitch_count, strikes, f'{pitch_count}-{strikes}' if pitch_count or strikes else '-'


def mlb_player_row(player, use_season_stats=False):
    source = player.get('season_stats') if use_season_stats else player.get('stats')
    source = source or {}
    batting = source.get('batting', {}) or {}
    pitching = source.get('pitching', {}) or {}
    position = get_path(player, 'position', 'abbreviation', default='') or ''
    is_pitcher = position == 'P' or has_stat_values(pitching) and not has_stat_values(batting)
    innings = pitching.get('inningsPitched') or ''
    earned_runs = int_or_zero(pitching.get('earnedRuns'))
    pitch_count, pitch_strikes, pc_st = pitch_count_strikes(pitching)

    return {
        'playerId': get_path(player, 'person', 'id', default=get_path(player, 'person', 'full_name', default='Unknown Player')),
        'name': get_path(player, 'person', 'full_name', default='Unknown Player'),
        'position': position,
        'role': 'P' if is_pitcher else 'B',
        'summary': (batting.get('summary') or pitching.get('summary') or '').strip(),
        'battingLine': batting_line(batting),
        'runs': int_or_zero(batting.get('runs')),
        'rbi': int_or_zero(batting.get('rbi')),
        'homeRuns': int_or_zero(batting.get('homeRuns')),
        'walks': int_or_zero(batting.get('baseOnBalls')),
        'strikeOuts': int_or_zero(batting.get('strikeOuts')),
        'avg': batting.get('avg') or '.000',
        'ops': batting.get('ops') or '.000',
        'inningsPitched': innings,
        'earnedRuns': earned_runs,
        'pitchingStrikeOuts': int_or_zero(pitching.get('strikeOuts')),
        'pitchingWalks': int_or_zero(pitching.get('baseOnBalls')),
        'era': calculated_era(earned_runs, innings, pitching.get('era')),
        'pitchCount': pitch_count,
        'pitchStrikes': pitch_strikes,
        'pitchCountStrikes': pc_st,
    }


def mlb_player_sort_key(player):
    if player['role'] == 'P':
        return (0, -parse_number(player['inningsPitched']), -player['pitchingStrikeOuts'], player['earnedRuns'])
    return (
        1,
        -player['homeRuns'],
        -player['rbi'],
        -parse_number(player['battingLine'].split('-')[0]),
        -player['runs'],
        -parse_number(player['ops']),
    )


def team_mlb_players(box_score, side, has_started):
    team = box_team(box_score, side)
    players = team.get('players', {})
    ordered_ids = [f'ID{player_id}' for player_id in team.get('pitchers', []) + team.get('batters', [])]
    ordered_ids.extend(player_id for player_id in players if player_id not in ordered_ids)
    rows = []
    seen_player_ids = set()

    for player_id in ordered_ids:
        player = players.get(player_id)
        if not player:
            continue
        person_id = get_path(player, 'person', 'id', default=player_id)
        if person_id in seen_player_ids:
            continue
        seen_player_ids.add(person_id)
        if has_started:
            stats = player.get('stats', {}) or {}
            if not has_stat_values(stats.get('batting', {}) or {}) and not has_stat_values(stats.get('pitching', {}) or {}):
                continue
        row = mlb_player_row(player, use_season_stats=not has_started)
        if row['role'] == 'B' and row['battingLine'] == '0-0' and not has_started:
            continue
        rows.append(row)

    if has_started:
        return sorted(rows, key=mlb_player_sort_key)[:12]
    return sorted(rows, key=lambda row: (parse_number(row['ops']), row['homeRuns'], row['rbi']), reverse=True)[:10]


def mlb_game_player_stats(game_id):
    box_score = dump_model(mlb_client().get_game_box_score(int(game_id)))
    line_score = dump_model(mlb_client().get_game_line_score(int(game_id)))
    players = []
    for side in TEAM_SIDES:
        players.extend((box_team(box_score, side).get('players', {}) or {}).values())
    has_started = int_or_zero(line_score.get('current_inning')) > 0 or any(
        has_stat_values((player.get('stats', {}) or {}).get('batting', {}) or {})
        or has_stat_values((player.get('stats', {}) or {}).get('pitching', {}) or {})
        for player in players
    )
    return {
        'hasStarted': has_started,
        'sport': 'mlb',
        'away': team_mlb_players(box_score, 'away', has_started),
        'home': team_mlb_players(box_score, 'home', has_started),
    }
