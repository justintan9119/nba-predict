from flask import Flask, jsonify, request
from flask_cors import CORS
from nba_api.stats.static import players
from nba_api.stats.endpoints import scoreboardv3, boxscoretraditionalv3
from nba_api.live.nba.endpoints import scoreboard
import datetime
from predict import train


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type"]}})
# @app.route("/api/players", methods=['GET'])
# def get_players():
#     active_players = players.get_active_players()
#     return jsonify({
#         'status': 'success', 
#         'players': active_players[:5]})

@app.route("/api/scoreboard", methods=['GET'])
def get_scoreboard():
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    today = "2026-6-5"
    board = scoreboardv3.ScoreboardV3(game_date = today)
    games = board.get_dict()['scoreboard']['games']

    all_games = [
        {
            'gameId': game['gameId'], 
            'away': game['awayTeam']['teamName'],
            'home': game['homeTeam']['teamName']
        }
        for game in games
    ]

    return jsonify({
        'date': today, 
        'teams': all_games,
    })

@app.route("/api/players/<game_id>", methods=['GET'])
def get_players(game_id):
    box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    awayTeam = box.get_dict()["boxScoreTraditional"]['awayTeam']
    homeTeam = box.get_dict()["boxScoreTraditional"]['homeTeam']
    
    print(homeTeam)
    return jsonify({
        'status': 'success',
    })

@app.route("/api/predict/<game_id>", methods=['GET'])
def get_id(game_id):
    box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    awayTeam = box.get_dict()["boxScoreTraditional"]['awayTeam']["teamId"]
    homeTeam = box.get_dict()["boxScoreTraditional"]['homeTeam']["teamId"]
    winner, confidence = train(homeTeam, awayTeam)
    confidence = round(confidence, 2)
    return jsonify({
        'status': 'success',
        'winner': winner,
        'conf': confidence

    })

@app.route("/api/game-info", methods=['POST'])
def game_info():
    data = request.json
    return jsonify({
        'status': 'success', 
    })

if __name__ == "__main__":
    app.run(debug=True, port=8080)
