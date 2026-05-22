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
        df_game = leaguegamelog.LeagueGameLog(season='2025-26').get_data_frames()[0]
    except:
        df_game = leaguegamelog.LeagueGameLog(season='2024-25').get_data_frames()[0]

    X = []
    y = []



if __name__ == "__main__":
    
