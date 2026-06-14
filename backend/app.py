from flask import Flask, jsonify, request
from flask_cors import CORS
from nba_api.stats.static import players
from nba_api.stats.endpoints import scoreboardv3, boxscoresummaryv3
from nba_api.live.nba.endpoints import scoreboard
import datetime
from predict import train, predict_live


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type"]}})

@app.route("/api/scoreboard", methods=['GET'])
def get_scoreboard():
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    # today = "2026-6-12"
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
    box = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id)
    awayTeam = box.get_dict()["boxScoreSummary"]['awayTeamId']
    homeTeam = box.get_dict()["boxScoreSummary"]['homeTeamId']
    
    print(homeTeam)
    return jsonify({
        'status': 'success',
    })

@app.route("/api/predict/<game_id>", methods=['GET'])
def get_id(game_id):
    box = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id)
    awayTeam = box.get_dict()["boxScoreSummary"]["awayTeamId"]
    homeTeam = box.get_dict()["boxScoreSummary"]["homeTeamId"]
    winner, confidence = train(homeTeam, awayTeam)
    confidence = round(confidence, 2)
    return jsonify({
        'status': 'success',
        'winner': winner,
        'conf': confidence

    })

@app.route("/api/live/<game_id>", methods=['GET'])
def get_live_prediction(game_id):
    try:
        prediction = predict_live(game_id)
        return jsonify({
            'status': 'success',
            'data': prediction
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route("/api/train", methods=['GET'])
def silent_train():
    train(None, None)
    return jsonify({'status': 'trained'})

@app.route("/api/game-info", methods=['POST'])
def game_info():
    data = request.json
    return jsonify({
        'status': 'success', 
    })

if __name__ == "__main__":
    app.run(debug=True, port=8080)
