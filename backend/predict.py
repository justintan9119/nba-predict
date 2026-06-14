import pandas as pd
import requests
import re
from nba_api.stats.endpoints import leaguedashteamstats, leaguegamelog
from nba_api.live.nba.endpoints import boxscore
from sklearn.ensemble import RandomForestClassifier
import time
import datetime

def get_nba_seasons(num_seasons):
    now = datetime.datetime.now()
    current_year = now.year
    if now.month < 10:
        current_year -= 1
    seasons = []
    for i in range(num_seasons):
        start_year = current_year - i
        end_year = (start_year + 1) % 100
        season_str = f"{start_year}-{end_year:02d}"
        seasons.append(season_str)
    return seasons[::-1]

_MODEL_CACHE = None
_FEATURES_CACHE = None

def train(team_id1, team_id2):
    global _MODEL_CACHE, _FEATURES_CACHE
    if _MODEL_CACHE is None:
        num_seasons = 5
        import datetime
        now = datetime.datetime.now()

        num_seasons = 5
        is_postseason = now.month in [4, 5, 6]
        if is_postseason:
            s_types = ['Playoffs', 'PlayIn']
            num_seasons = 10
        else:
            s_types = ['Regular Season', 'Playoffs', 'PlayIn']

        seasons = get_nba_seasons(num_seasons)
        all_seasons_games = []

        print(f"--- Training model ---")
        for season in seasons:
            try:
                games_season = leaguegamelog.LeagueGameLog(season=season).get_data_frames()[0]
                all_seasons_games.append(games_season)
                time.sleep(0.6)
            except: continue

        games = pd.concat(all_seasons_games, ignore_index=True)
        
        games['POSS'] = games['FGA'] + 0.44 * games['FTA'] - games['OREB'] + games['TOV']
        games['OFF_RATING'] = (games['PTS'] / games['POSS']) * 100
        games['TS_PCT'] = games['PTS'] / (2 * (games['FGA'] + 0.44 * games['FTA']))
        games['AST_100'] = (games['AST'] / games['POSS']) * 100
        games['REB_100'] = (games['REB'] / games['POSS']) * 100
        games['TOV_100'] = (games['TOV'] / games['POSS']) * 100
        games['STL_100'] = (games['STL'] / games['POSS']) * 100
        games['BLK_100'] = (games['BLK'] / games['POSS']) * 100
        
        features_to_diff = ['OFF_RATING', 'TS_PCT', 'AST_100', 'REB_100', 'TOV_100', 'STL_100', 'BLK_100']
        opp_stats = games[['GAME_ID', 'TEAM_ID'] + features_to_diff].copy()
        opp_stats.columns = ['GAME_ID', 'OPP_TEAM_ID'] + [f'OPP_{c}' for c in features_to_diff]
        
        merged = games.merge(opp_stats, on='GAME_ID')
        merged = merged[merged['TEAM_ID'] != merged['OPP_TEAM_ID']]
        
        merged['IS_HOME'] = merged['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
        
        diff_data = pd.DataFrame()
        diff_data['WL'] = merged['WL'].map({'W': 1, 'L': 0})
        diff_data['IS_HOME'] = merged['IS_HOME']

        for col in features_to_diff:
            diff_data[col] = merged[col] - merged[f'OPP_{col}']
            
        diff_data = diff_data.dropna()
        X = diff_data.drop(columns=['WL'])
        y = diff_data['WL']

        model = RandomForestClassifier( n_estimators = 100, 
                                random_state = 42, 
                                max_depth = 12, 
                                min_samples_leaf = 10, 
                                max_features = 'sqrt')

        model.fit(X, y)
        _MODEL_CACHE, _FEATURES_CACHE = model, X.columns.tolist()
        print(f"--- Training Complete ---")

    if team_id1 is None: return None, None
    return match_up(_MODEL_CACHE, team_id1, team_id2, _FEATURES_CACHE)

def match_up(model, team1_id, team2_id, feature_columns):
    t1_id, t2_id = int(team1_id), int(team2_id)
    
    def get_stats(season_year=None):
        params = {'measure_type_detailed_defense': 'Base'}
        adv_params = {'measure_type_detailed_defense': 'Advanced'}
        if season_year:
            params['season'] = season_year
            adv_params['season'] = season_year
            
        base = leaguedashteamstats.LeagueDashTeamStats(**params).get_data_frames()[0]
        adv = leaguedashteamstats.LeagueDashTeamStats(**adv_params).get_data_frames()[0]
        return pd.merge(base, adv, on=['TEAM_ID', 'TEAM_NAME'], suffixes=('', '_adv'))

    stats = get_stats()

    if stats.empty:
        last_season = get_nba_seasons(2)[0] 
        stats = get_stats(season_year=last_season)
 
    stats['AST_100'] = (stats['AST'] / stats['POSS']) * 100
    stats['REB_100'] = (stats['REB'] / stats['POSS']) * 100
    stats['TOV_100'] = (stats['TOV'] / stats['POSS']) * 100
    stats['STL_100'] = (stats['STL'] / stats['POSS']) * 100
    stats['BLK_100'] = (stats['BLK'] / stats['POSS']) * 100
    
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
    
    game_diff = pd.DataFrame(team1_stats.values - team2_stats.values, columns=stat_features)
    game_diff['IS_HOME'] = 1 
    game_diff = game_diff[feature_columns] # Ensure correct column order
    
    prediction = model.predict(game_diff)
    probability = model.predict_proba(game_diff)[0]

    team1_name = stats[stats['TEAM_ID'] == t1_id]['TEAM_NAME'].values[0]
    team2_name = stats[stats['TEAM_ID'] == t2_id]['TEAM_NAME'].values[0]

    winner = team1_name if prediction[0] == 1 else team2_name
    confidence = max(probability) * 100

    return winner, confidence

def predict_live(game_id):
    if _MODEL_CACHE is None: train(None, None)
    
    url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
    res = requests.get(url, headers={'Referer': 'https://www.nba.com/', 'User-Agent': 'Mozilla/5.0'})
    if res.status_code != 200:
        raise Exception(f"Failed to fetch live data: {res.status_code}")
    box = res.json()['game']
    
    if box['gameStatus'] == 1: raise Exception("Game has not started yet")

    is_final = box['gameStatus'] == 3
    home_name = box['homeTeam']['teamName']
    away_name = box['awayTeam']['teamName']
    home_score = box['homeTeam']['score']
    away_score = box['awayTeam']['score']

    # Treat as final if official status is 3 OR if clock is 0:00 in 4th+ and not tied
    clock_match = re.search(r'PT(\d+)M(\d+)', box['gameClock'])
    mins = int(clock_match.group(1)) if clock_match else 0
    secs = int(clock_match.group(2)) if clock_match else 0

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

    def get_stats(team):
        s = team['statistics']
        p = s['fieldGoalsAttempted'] + 0.44 * s['freeThrowsAttempted'] - s['reboundsOffensive'] + s['turnovers'] or 1
        return pd.Series({
            'OFF_RATING': (team['score'] / p) * 100,
            'TS_PCT': team['score'] / (2 * (s['fieldGoalsAttempted'] + 0.44 * s['freeThrowsAttempted'])) if (s['fieldGoalsAttempted'] + 0.44 * s['freeThrowsAttempted']) > 0 else 0,
            'AST_100': (s['assists'] / p) * 100,
            'REB_100': (s['reboundsTotal'] / p) * 100,
            'TOV_100': (s['turnovers'] / p) * 100,
            'STL_100': (s['steals'] / p) * 100,
            'BLK_100': (s['blocks'] / p) * 100
        })

    h_stats = get_stats(box['homeTeam'])
    a_stats = get_stats(box['awayTeam'])
    
    # print(f"\n--- LIVE STATS ({home_name} vs {away_name}) ---")
    # print(f"{home_name} (Home):\n{h_stats.to_dict()}")
    # print(f"{away_name} (Away):\n{a_stats.to_dict()}")

    diff = (h_stats - a_stats).to_frame().T
    diff['IS_HOME'] = 1
    diff = diff[_FEATURES_CACHE]
    
    ml_prob = _MODEL_CACHE.predict_proba(diff)[0]
    ml_winner_idx = _MODEL_CACHE.predict(diff)[0]

    #clock
    clock_match = re.search(r'PT(\d+)M(\d+)', box['gameClock'])
    mins = int(clock_match.group(1)) if clock_match else 0
    secs = int(clock_match.group(2)) if clock_match else 0
    remaining_in_period = mins * 60 + secs
    periods_left = max(0, 4 - box['period'])
    total_remaining = (periods_left * 720) + remaining_in_period
    
    time_factor = 1 - (total_remaining / 2880)
    
    margin = home_score - away_score
    # A 10 point lead with 0 time left is 100%. A 2 point lead with 1 min left is high.
    # Adjust sensitivity based on time
    sensitivity = 0.1 + (time_factor * 0.5) 
    import math
    score_prob = 1 / (1 + math.exp(-margin * sensitivity))
    
    # 4. Blend Projections
    # Early game is 90% ML efficiency. Late game is 95% Scoreboard.
    weight = 0.1 + (time_factor * 0.85)
    final_home_prob = (ml_prob[1] * (1 - weight)) + (score_prob * weight)
    
    winner = home_name if final_home_prob > 0.5 else away_name
    confidence = final_home_prob if final_home_prob > 0.5 else (1 - final_home_prob)

    clock = box['gameClock']
    if clock_match:
        clock = f"{mins}:{secs:02d}"
    
    return {
        'winner': winner,
        'confidence': round(confidence * 100, 2),
        'home': home_name,
        'away': away_name,
        'score': f"{away_score} - {home_score}",
        'clock': clock,
        'period': box['period']
    }
