import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from nba_api.stats.endpoints import leaguegamelog
import numpy as np
import os

df_stats = pd.read_csv('teams_stats.csv', encoding='latin-1')

features = [
    'OFF_RATING', 
    'DEF_RATING', 
    'NET_RATING', 
    'EFG_PCT', 
    'TS_PCT',
    'REB_PCT',
    'TM_TOV_PCT'
]

def train_model():

    try:
        game_log = leaguegamelog.LeagueGameLog(season='2025-26').get_data_frames()[0]
    except:
   
        game_log = leaguegamelog.LeagueGameLog(season='2024-25').get_data_frames()[0]

    X = []
    y = []

    # 3. Create the Training Set
    # We loop through every game and look at the stat difference between the two teams
    # Each game ID appears twice (once for each team), so we look at every 2nd row
    for i in range(0, len(game_log), 2):
        game_row = game_log.iloc[i]
        
        # Determine which teams were playing
        # Matchup format is usually "BOS vs. NYK"
        matchup = game_row['MATCHUP']
        team1_name = game_row['TEAM_NAME']
        
        # Find the opponent
        # If the matchup is "BOS vs. NYK", and team1 is Boston, team2 is New York
        # We find the other row with the same GAME_ID
        game_id = game_row['GAME_ID']
        opponent_row = game_log[(game_log['GAME_ID'] == game_id) & (game_log['TEAM_NAME'] != team1_name)].iloc[0]
        team2_name = opponent_row['TEAM_NAME']

        # Get the stats for both teams from our CSV data
        t1_stats_row = df_stats[df_stats['TEAM_NAME'] == team1_name]
        t2_stats_row = df_stats[df_stats['TEAM_NAME'] == team2_name]

        if not t1_stats_row.empty and not t2_stats_row.empty:
            # Calculate the difference
            t1_vals = t1_stats_row.iloc[0][features]
            t2_vals = t2_stats_row.iloc[0][features]
            
            diff = (t1_vals - t2_vals).astype(float)
            X.append(diff)
            
            # The target (y) is 1 if Team 1 won, 0 if they lost
            y.append(1 if game_row['WL'] == 'W' else 0)

    # 4. Train the Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Train the model globally
model = train_model()

def predict_winner(team1_name, team2_name):

    try:
        t1_stats = df_stats[df_stats['TEAM_NAME'] == team1_name].iloc[0][features]
        t2_stats = df_stats[df_stats['TEAM_NAME'] == team2_name].iloc[0][features]
        
        diff = (t1_stats - t2_stats).values.reshape(1, -1)
        
        prediction = model.predict(diff)[0]
        prob = model.predict_proba(diff)[0]
        
        if prediction:
            winner = team1_name
            confidence = prob[1]
        else:
            winner = team2_name
            confidence = prob[0]

        return winner, confidence
    except Exception as e:
        return f"Error: {e}", 0

# if __name__ == "__main__":
#     # Test a prediction
#     t1, t2 = "Boston Celtics", "Milwaukee Bucks"
#     winner, conf = predict_winner(t1, t2)
#     print(f"\nMatchup: {t1} vs {t2}")
#     print(f"Predicted Winner: {winner} ({conf:.2%} confidence)")
