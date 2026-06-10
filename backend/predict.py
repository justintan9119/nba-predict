import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, teamgamelog, leaguegamelog, boxscoresummaryv3
from nba_api.stats.static import teams
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
import time

def get_nba_seasons(num_seasons):
    import datetime
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

def train(team_id1, team_id2):
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

    for season in seasons:
        for s_type in s_types:
            try:
                games_season = leaguegamelog.LeagueGameLog(season = season, season_type_all_star=s_type).get_data_frames()[0]
                all_seasons_games.append(games_season)
                time.sleep(0.6)
            except:
                continue

    games = pd.concat(all_seasons_games, ignore_index=True)
    
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values('GAME_DATE')


    opp_stats = games[['GAME_ID', 'TEAM_ID', 'PTS', 'FGM', 'FGA', 'FG3M', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']].copy()
    opp_stats.columns = ['GAME_ID', 'OPP_TEAM_ID', 'OPP_PTS', 'OPP_FGM', 'OPP_FGA', 'OPP_FG3M', 'OPP_FTA', 'OPP_OREB', 'OPP_DREB', 'OPP_REB', 'OPP_AST', 'OPP_STL', 'OPP_BLK', 'OPP_TOV', 'OPP_PF']
    
    games = games.merge(opp_stats, on='GAME_ID')
    games = games[games['TEAM_ID'] != games['OPP_TEAM_ID']]

    games['POSS'] = games['FGA'] + 0.44 * games['FTA'] - games['OREB'] + games['TOV']
    games['OPP_POSS'] = games['OPP_FGA'] + 0.44 * games['OPP_FTA'] - games['OPP_OREB'] + games['OPP_TOV']
    
    games['OFF_RATING'] = (games['PTS'] / games['POSS']) * 100
    games['DEF_RATING'] = (games['OPP_PTS'] / games['OPP_POSS']) * 100
    games['TS_PCT'] = games['PTS'] / (2 * (games['FGA'] + 0.44 * games['FTA']))
    games['EFG_PCT'] = (games['FGM'] + 0.5 * games['FG3M']) / games['FGA']
    games['AST_RATIO'] = (games['AST'] * 100) / games['POSS']
    games['REB_PCT'] = games['REB'] / (games['REB'] + games['OPP_REB'])
    
    games['PIE'] = (games['PTS'] + games['FGM'] + games['FTM'] - games['FGA'] - games['FTA'] + 
                    games['DREB'] + (0.5 * games['OREB']) + games['AST'] + games['STL'] + 
                    (0.5 * games['BLK']) - games['PF'] - games['TOV'])

    features_to_keep = ['WL', 'OFF_RATING', 'DEF_RATING', 'TS_PCT', 'EFG_PCT', 'AST_RATIO', 'REB_PCT', 'PIE']
    games = games[features_to_keep]
    
    games["WL"] = games["WL"].map({'W':1, 'L':0})
    games = games.dropna(subset=['WL'])

    X = games.drop(columns=['WL'])
    y = games['WL']

    tscv = TimeSeriesSplit(n_splits=5)
    
    model = RandomForestClassifier( n_estimators = 100, 
                                    random_state = 42, 
                                    max_depth = 12, 
                                    min_samples_leaf = 10, 
                                    max_features = 'sqrt')

    model.fit(X, y)
    feature_order = X.columns.tolist()

    return match_up(model, team_id1, team_id2, feature_order)

def match_up(model, team1_id, team2_id, feature_columns):
    
    base_stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Base').get_data_frames()[0]
    adv_stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
    
    stats = pd.merge(base_stats, adv_stats, on=['TEAM_ID', 'TEAM_NAME'], suffixes=('', '_adv'))
    
    team1_stats = stats[stats['TEAM_ID'] == team1_id][feature_columns]
    team2_stats = stats[stats['TEAM_ID'] == team2_id][feature_columns]
    
    game_diff = pd.DataFrame(team1_stats.values - team2_stats.values, columns=feature_columns)

    prediction = model.predict(game_diff)
    probability = model.predict_proba(game_diff)[0]

    team1_name = stats[stats['TEAM_ID'] == team1_id]['TEAM_NAME'].values[0]
    team2_name = stats[stats['TEAM_ID'] == team2_id]['TEAM_NAME'].values[0]

    winner = team1_name if prediction[0] == 1 else team2_name
    confidence = max(probability) * 100

    return winner, confidence
