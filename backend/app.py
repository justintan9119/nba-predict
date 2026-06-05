from flask import Flask, jsonify, request
from flask_cors import CORS
from nba_api.stats.static import players
from nba_api.stats.endpoints import scoreboardv3
from nba_api.live.nba.endpoints import scoreboard
import datetime


app = Flask(__name__)
CORS(app)

# @app.route("/api/players", methods=['GET'])
# def get_players():
#     active_players = players.get_active_players()
#     return jsonify({
#         'status': 'success', 
#         'players': active_players[:5]})

@app.route("/api/scoreboard", methods=['GET'])
def get_scoreboard():
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    board = scoreboardv3.ScoreboardV3(game_date = "2026-3-5")
    games = board.get_dict()['scoreboard']['games']
    
    all_games = [
        [game['awayTeam']['teamName'],game['homeTeam']['teamName']]
        for game in games
    ]

    return jsonify({
        'date': today, 
        'teams': all_games
    })

# @app.route("/api/game-info", methods=['POST'])
# def game_info():
#     data = request.json
#     return jsonify({
#         'status': 'success', 
#     })

if __name__ == "__main__":
    app.run(debug=True, port=8080)
