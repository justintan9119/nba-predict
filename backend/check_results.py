from nba_api.stats.endpoints import leaguegamelog
import pandas as pd

def get_season_results():
    # LeagueGameLog retrieves results for every game in a specific season
    game_log = leaguegamelog.LeagueGameLog(season='2024-25', season_type_all_star='Regular Season')
    df = game_log.get_data_frames()[0]
    
    # Each game has two rows (one for each team). 
    # We can filter to see the winner, scores, etc.
    print(df[['GAME_ID', 'TEAM_NAME', 'MATCHUP', 'WL', 'PTS']].head(10))
    return df

if __name__ == "__main__":
    try:
        results = get_season_results()
        print(f"\nSuccessfully retrieved {len(results)} team-game entries.")
    except Exception as e:
        print(f"Error: {e}")
