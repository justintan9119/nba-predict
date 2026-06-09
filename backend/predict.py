
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, teamgamelog, leaguegamelog, boxscoresummaryv3
from nba_api.stats.static import teams
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import time
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
def get_nba_seasons(num_seasons = 10):
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
def train(team_id1, team_id2, num_seasons = 5):

    seasons = get_nba_seasons(num_seasons)
    all_seasons_games = []
    
    for season in seasons:
        for s_type in ['Regular Season', 'Playoffs', 'PlayIn']:
            games_season = leaguegamelog.LeagueGameLog(season = season, season_type_all_star=s_type).get_data_frames()[0]
            all_seasons_games.append(games_season)
            time.sleep(0.5)

    games = pd.concat(all_seasons_games, ignore_index=True)
    
    game_ids = games['GAME_ID'].unique()
    games = games.drop(columns = ['SEASON_ID', 'TEAM_ABBREVIATION', 'VIDEO_AVAILABLE', 'PLUS_MINUS', 'MIN',
                                  'GAME_ID', 'GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'TEAM_ID'])
    games["WL"] = games["WL"].map({'W':1, 'L':0})
    games = games.dropna(subset=['WL'])

    X = games.iloc[:, 1:]
    y = games.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.25)

    model = RandomForestClassifier( n_estimators = 100, 
                                    random_state = 42, 
                                    max_depth = 12, 
                                    min_samples_leaf = 10, 
                                    max_features = 'sqrt')
    model.fit(X, y)
    # y_pred = model.predict(X_test)
    # print(model.score(X_test,y_test))
    # print(classification_report(y_test, y_pred))
    # features = pd.DataFrame(model.feature_importances_, index = X.columns)
    # print(features.head(20))
    return match_up(model, team_id1, team_id2, X.columns)

def match_up(model, team1_id, team2_id, feature_columns):
    
    stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Base').get_data_frames()[0]

    # Filter stats for the specific teams and columns
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
    
# if __name__ == "__main__":
#     print(train(1610612744, 1610612747))