import env  # noqa: F401 - loads backend/.env before local imports read os.environ

from flask import Flask, jsonify, request
from flask_cors import CORS
from nba_api.stats.endpoints import scoreboardv3, boxscoresummaryv3
from predict import format_clock, model_diagnostics, predict_live, predict_matchup_details, start_training
from odds import fetch_moneyline_odds
from kalshi_trading import kalshi_record_summary, kalshi_user_bets_for_odds, maybe_place_edge_bet
from live_stats import live_game_stats
from mlb_players import mlb_game_player_stats
from mlb_predict import mlb_live_projection, mlb_model_diagnostics, mlb_predict_game, mlb_scoreboard_games, start_mlb_training
import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type"]}})

LEAGUE_IDS = {
    'nba': '00',
    'wnba': '10',
}
VALID_LEAGUES = set(LEAGUE_IDS) | {'mlb'}


def success_response(**data):
    return jsonify({'status': 'success', **data})


def get_league():
    league = request.args.get('league', 'nba').lower()
    return league if league in VALID_LEAGUES else 'nba'


def get_scoreboard_date():
    selected_date = request.args.get('date') or datetime.datetime.now().strftime('%Y-%m-%d')
    try:
        datetime.datetime.strptime(selected_date, '%Y-%m-%d')
        return selected_date
    except ValueError:
        return datetime.datetime.now().strftime('%Y-%m-%d')


def get_game_team_ids(game_id):
    summary = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id).get_dict()["boxScoreSummary"]
    return summary["homeTeamId"], summary["awayTeamId"]


def full_team_name(team):
    city = team.get('teamCity') or team.get('teamCityName') or ''
    name = team.get('teamName') or ''
    return f"{city} {name}".strip() or name


def team_logo_url(league, team_id):
    return f'https://cdn.nba.com/logos/{league}/{team_id}/global/L/logo.svg'


def scheduled_start_time(game):
    return (
        game.get('gameTimeUTC')
        or game.get('gameTimeEst')
        or game.get('gameTimeEastern')
        or game.get('gameEt')
        or game.get('gameTime')
        or game.get('gameDateTimeUTC')
    )


def game_payload(game, league):
    away_team = game['awayTeam']
    home_team = game['homeTeam']
    return {
        'gameId': game['gameId'],
        'away': away_team['teamName'],
        'home': home_team['teamName'],
        'awayFullName': full_team_name(away_team),
        'homeFullName': full_team_name(home_team),
        'awayTeamId': away_team['teamId'],
        'homeTeamId': home_team['teamId'],
        'awayLogo': team_logo_url(league, away_team['teamId']),
        'homeLogo': team_logo_url(league, home_team['teamId']),
        'awayScore': away_team.get('score', 0),
        'homeScore': home_team.get('score', 0),
        'status': game.get('gameStatus'),
        'statusText': game.get('gameStatusText', ''),
        'startTime': scheduled_start_time(game),
        'clock': format_clock(game.get('gameClock', '')),
        'period': game.get('period', 0),
        'isLive': game.get('gameStatus') == 2,
        'isFinal': game.get('gameStatus') == 3 or game.get('gameStatusText', '').lower() == 'final'
    }


def scoreboard_games(league, selected_date):
    if league == 'mlb':
        return mlb_scoreboard_games(selected_date)
    board = scoreboardv3.ScoreboardV3(game_date=selected_date, league_id=LEAGUE_IDS[league])
    return [game_payload(game, league) for game in board.get_dict()['scoreboard']['games']]


def find_scoreboard_game(game_id, league, selected_date):
    return next((game for game in scoreboard_games(league, selected_date) if game['gameId'] == game_id), None)


@app.route("/api/scoreboard", methods=['GET'])
def get_scoreboard():
    league = get_league()
    selected_date = get_scoreboard_date()
    all_games = scoreboard_games(league, selected_date)

    return jsonify({
        'league': league,
        'date': selected_date,
        'teams': all_games,
    })


@app.route("/api/odds/<game_id>", methods=['GET'])
def get_moneyline_odds(game_id):
    league = get_league()
    selected_date = get_scoreboard_date()
    game = find_scoreboard_game(game_id, league, selected_date)
    if not game:
        return jsonify({'status': 'error', 'message': 'Game was not found on the selected scoreboard date.'}), 404
    try:
        odds = fetch_moneyline_odds(league, game)
        if odds:
            try:
                odds['userBets'] = kalshi_user_bets_for_odds(odds)
            except Exception as error:
                odds['userBets'] = []
                odds['userBetsError'] = str(error)
        return success_response(odds=odds)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 503


@app.route("/api/kalshi/edge-bet/<game_id>", methods=['POST'])
def place_kalshi_edge_bet(game_id):
    league = get_league()
    selected_date = get_scoreboard_date()
    game = find_scoreboard_game(game_id, league, selected_date)
    if not game:
        return jsonify({'status': 'error', 'message': 'Game was not found on the selected scoreboard date.'}), 404
    try:
        if league == 'mlb':
            prediction = mlb_predict_game(game_id, selected_date)
        else:
            home_team_id, away_team_id = get_game_team_ids(game_id)
            prediction = predict_matchup_details(home_team_id, away_team_id, league)
        odds = fetch_moneyline_odds(league, game)
        result = maybe_place_edge_bet(game, odds, prediction.get('probabilities'))
        return success_response(result=result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 503


@app.route("/api/kalshi/record", methods=['GET'])
def get_kalshi_record():
    try:
        return success_response(record=kalshi_record_summary(request.args.get('startDate')))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 503

@app.route("/api/players/<game_id>", methods=['GET'])
def get_players(game_id):
    league = get_league()
    if league == 'mlb':
        try:
            return success_response(**mlb_game_player_stats(game_id))
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 503
    try:
        return success_response(**live_game_stats(
            game_id,
            league,
            request.args.get('date'),
            request.args.get('awayTeam'),
            request.args.get('homeTeam'),
            request.args.get('awayTeamId'),
            request.args.get('homeTeamId'),
        ))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 503

@app.route("/api/predict/<game_id>", methods=['GET'])
def get_id(game_id):
    league = get_league()
    try:
        if league == 'mlb':
            prediction = mlb_predict_game(game_id, request.args.get('date'))
        else:
            home_team_id, away_team_id = get_game_team_ids(game_id)
            prediction = predict_matchup_details(home_team_id, away_team_id, league)
        return success_response(
            winner=prediction['winner'],
            conf=round(prediction['confidence'], 2),
            probabilities={side: round(value, 4) for side, value in prediction['probabilities'].items()},
            modelProbabilities={side: round(value, 4) for side, value in prediction['modelProbabilities'].items()},
            metrics=prediction.get('metrics'),
            analysis=prediction.get('analysis'),
        )
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 503

@app.route("/api/model-diagnostics", methods=['GET'])
def get_model_diagnostics():
    if get_league() == 'mlb':
        diagnostics = mlb_model_diagnostics()
        if diagnostics is None:
            return jsonify({'status': 'error', 'message': 'MLB model diagnostics are unavailable until training completes.'}), 503
        return success_response(diagnostics=diagnostics)
    diagnostics = model_diagnostics(get_league())
    if diagnostics is None:
        return jsonify({'status': 'error', 'message': 'Model diagnostics are unavailable until training completes.'}), 503
    return success_response(diagnostics=diagnostics)

@app.route("/api/live/<game_id>", methods=['GET'])
def get_live_prediction(game_id):
    league = get_league()
    try:
        prediction = mlb_live_projection(game_id) if league == 'mlb' else predict_live(game_id, league)
        return success_response(data=prediction)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route("/api/train", methods=['GET'])
def silent_train():
    league = get_league()
    if league == 'mlb':
        return jsonify({'status': start_mlb_training(), 'league': league})
    return jsonify({'status': start_training(league), 'league': league})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
