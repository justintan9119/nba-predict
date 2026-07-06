import env  # noqa: F401 - loads backend/.env before reading configuration

import os
import re
import ssl

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_ODDS_API_KEY = '6fc82da43ff57db0ba1c60c8199b567d'


def configured_odds_api_key():
    api_key = os.environ.get('ODDS_API_KEY', '').strip()
    if not api_key or api_key.lower() in {'your_api_key', 'your-api-key', 'replace_me'}:
        return DEFAULT_ODDS_API_KEY
    return api_key


ODDS_API_KEY = configured_odds_api_key()
SPORT_KEYS = {
    'nba': 'basketball_nba',
    'wnba': 'basketball_wnba',
}
ODDS_API_URL = 'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/'


class TLS12Adapter(HTTPAdapter):
    """Use TLS 1.2 for the Odds API's CloudFront endpoint."""

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_2
        pool_kwargs['ssl_context'] = context
        return super().init_poolmanager(connections, maxsize, block, **pool_kwargs)


def create_odds_api_session():
    session = requests.Session()
    retries = Retry(
        total=2,
        connect=2,
        read=0,
        status=0,
        backoff_factor=0.25,
        allowed_methods=frozenset({'GET'}),
    )
    session.mount('https://api.the-odds-api.com/', TLS12Adapter(max_retries=retries))
    return session


ODDS_API_SESSION = create_odds_api_session()


def normalize_name(name):
    return re.sub(r'[^a-z0-9]+', '', str(name or '').lower())


def names_match(left, right):
    left_name = normalize_name(left)
    right_name = normalize_name(right)
    return left_name == right_name or left_name in right_name or right_name in left_name


def best_moneyline_from_event(event, home_name, away_name):
    for bookmaker in event.get('bookmakers', []):
        for market in bookmaker.get('markets', []):
            if market.get('key') != 'h2h':
                continue
            prices = {}
            for outcome in market.get('outcomes', []):
                outcome_name = outcome.get('name')
                if names_match(outcome_name, home_name):
                    prices['home'] = outcome.get('price')
                elif names_match(outcome_name, away_name):
                    prices['away'] = outcome.get('price')
            if 'home' in prices and 'away' in prices:
                return {
                    'home': prices['home'],
                    'away': prices['away'],
                    'bookmaker': bookmaker.get('key'),
                    'bookmakerTitle': bookmaker.get('title'),
                    'lastUpdate': bookmaker.get('last_update'),
                }
    return None


def find_matching_event(events, home_name, away_name):
    for event in events:
        event_home = event.get('home_team')
        event_away = event.get('away_team')
        if names_match(event_home, home_name) and names_match(event_away, away_name):
            return event
        if names_match(event_home, away_name) and names_match(event_away, home_name):
            return event
    return None


def fetch_moneyline_odds(league, game):
    sport_key = SPORT_KEYS.get(league, 'basketball_nba')
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
        'dateFormat': 'iso',
    }
    try:
        response = ODDS_API_SESSION.get(
            ODDS_API_URL.format(sport_key=sport_key),
            params=params,
            timeout=10,
        )
    except requests.RequestException as error:
        raise RuntimeError('Unable to reach the Odds API. Please try again shortly.') from error
    response.raise_for_status()
    events = response.json()
    home_name = game.get('homeFullName') or game.get('home')
    away_name = game.get('awayFullName') or game.get('away')
    event = find_matching_event(events, home_name, away_name)
    if not event:
        return None
    moneyline = best_moneyline_from_event(event, home_name, away_name)
    if not moneyline:
        return None
    return {
        'gameId': game.get('gameId'),
        'commenceTime': event.get('commence_time'),
        'bookmaker': moneyline.get('bookmaker'),
        'bookmakerTitle': moneyline.get('bookmakerTitle'),
        'lastUpdate': moneyline.get('lastUpdate'),
        'away': {
            'team': game.get('away'),
            'fullName': away_name,
            'price': moneyline['away'],
        },
        'home': {
            'team': game.get('home'),
            'fullName': home_name,
            'price': moneyline['home'],
        },
    }
