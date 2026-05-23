from flask import Flask, jsonify, request
from flask_cors import CORS
from nba_api.stats.static import players
from nba_api.stats.endpoints import scoreboardv3
import datetime
from predict_win import train_model, predict_winner

train_model()
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
    sb = scoreboardv3.ScoreboardV3(game_date = today)
    return jsonify({
        'status': 'success', 
        'games': sb.get_dict()['scoreboard']['games']
        })

@app.route("/api/game-info", methods=['POST'])
def receive_game_info():
    data = request.get_json()
    p1 = players.find_player_by_id(data.get('id1'))
    p2 = players.find_player_by_id(data.get('id2'))
    awayTeam = data.get('team1')
    homeTeam = data.get('team2')
    print("bro",awayTeam, homeTeam)
    winner, confidence = predict_winner(awayTeam, homeTeam)
    print("yo",winner, confidence)
    return jsonify({
        'status': 'success', 
        'names': [p1, p2],
        'teams': [awayTeam, homeTeam],
        'predict-info': [winner, f"{confidence:.2%}"]
        })



if __name__ == "__main__":
    app.run(debug=True, port=8080)
