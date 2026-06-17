from flask import Flask, jsonify, request
from flask_cors import CORS
from nba_api.stats.endpoints import scoreboardv3, boxscoresummaryv3
from predict import format_clock, train, predict_live, predict_matchup
import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type"]}})

SCOREBOARD_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
LEAGUE_IDS = {
    'nba': '00',
    'wnba': '10',
}


def success_response(**data):
    return jsonify({'status': 'success', **data})


def get_league():
    league = request.args.get('league', 'nba').lower()
    return league if league in LEAGUE_IDS else 'nba'


def get_game_team_ids(game_id):
    summary = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id).get_dict()["boxScoreSummary"]
    return summary["homeTeamId"], summary["awayTeamId"]


@app.route("/api/scoreboard", methods=['GET'])
def get_scoreboard():
    league = get_league()
    today = SCOREBOARD_DATE
    board = scoreboardv3.ScoreboardV3(game_date=today, league_id=LEAGUE_IDS[league])
    games = board.get_dict()['scoreboard']['games']

    all_games = [
        {
            'gameId': game['gameId'], 
            'away': game['awayTeam']['teamName'],
            'home': game['homeTeam']['teamName'],
            'awayScore': game['awayTeam'].get('score', 0),
            'homeScore': game['homeTeam'].get('score', 0),
            'status': game.get('gameStatus'),
            'statusText': game.get('gameStatusText', ''),
            'clock': format_clock(game.get('gameClock', '')),
            'period': game.get('period', 0),
            'isLive': game.get('gameStatus') == 2,
            'isFinal': game.get('gameStatus') == 3 or game.get('gameStatusText', '').lower() == 'final'
        }
        for game in games
    ]

    return jsonify({
        'league': league,
        'date': today, 
        'teams': all_games,
    })

@app.route("/api/players/<game_id>", methods=['GET'])
def get_players(game_id):
    home_team_id, away_team_id = get_game_team_ids(game_id)
    return success_response(homeTeamId=home_team_id, awayTeamId=away_team_id)

@app.route("/api/predict/<game_id>", methods=['GET'])
def get_id(game_id):
    league = get_league()
    try:
        home_team_id, away_team_id = get_game_team_ids(game_id)
        winner, confidence = predict_matchup(home_team_id, away_team_id, league)
        return success_response(winner=winner, conf=round(confidence, 2))
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 503

@app.route("/api/live/<game_id>", methods=['GET'])
def get_live_prediction(game_id):
    league = get_league()
    try:
        prediction = predict_live(game_id, league)
        return success_response(data=prediction)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route("/api/train", methods=['GET'])
def silent_train():
    league = get_league()
    train(None, None, league)
    return jsonify({'status': 'trained', 'league': league})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
