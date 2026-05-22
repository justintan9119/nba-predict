from flask import Flask, jsonify, request
from flask_cors import CORS
from nba_api.stats.static import players
from nba_api.stats.endpoints import scoreboardv3
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
    try:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        sb = scoreboardv3.ScoreboardV3(game_date=today)
        return jsonify({
            'status': 'success', 
            'games': sb.get_dict()['scoreboard']['games']
            })
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e)
            })

@app.route("/api/key-player-id", methods=['POST'])
def receive_player_ids():
    data = request.get_json()
    p1 = players.find_player_by_id(data.get('id1'))
    p2 = players.find_player_by_id(data.get('id2'))
    print(p1, p2)
    return jsonify({
        'status': 'success', 
        'names': [p1, p2]})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
