import csv
from nba_api.stats.endpoints import leaguedashteamstats

def generate_full_team_csv():  # Define the main function to fetch and save data

    stats = leaguedashteamstats.LeagueDashTeamStats(  # Call the NBA API to request team data
        measure_type_detailed_defense='Advanced', 
        season='2025-26', 
        per_mode_detailed='PerGame'
    )  
    
    data_dict = stats.get_dict()  # Convert the raw API response into a Python dictionary
    headers = data_dict['resultSets'][0]['headers']  # Extract the list of column names (headers)
    rows = data_dict['resultSets'][0]['rowSet']  # Extract the actual data for each team (rows)
    
    filename = 'teams_stats.csv' 
    
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:  # Open (or create) the CSV file for writing
        writer = csv.writer(output_file)  # Create a 'writer' object that helps format data for the CSV
        writer.writerow(headers)  
        writer.writerows(rows) 
    print("file made")
if __name__ == "__main__": 
    generate_full_team_csv()  
