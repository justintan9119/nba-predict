import env  # noqa: F401 - loads backend/.env before reading configuration

import datetime
import os
import re
import time
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

KALSHI_API_BASE = os.environ.get('KALSHI_API_BASE', 'https://external-api.kalshi.com/trade-api/v2').rstrip('/')
KALSHI_MAX_EVENT_PAGES = max(1, int(os.environ.get('KALSHI_MAX_EVENT_PAGES', '6')))
KALSHI_MAX_MARKET_PAGES = max(1, int(os.environ.get('KALSHI_MAX_MARKET_PAGES', '6')))
LEAGUE_COMPETITIONS = {
    'nba': {'Pro Basketball', 'Pro Basketball (M)', 'NBA'},
    'wnba': {'WNBA', 'Pro Basketball (W)'},
    'mlb': {'Pro Baseball', 'Baseball', 'MLB'},
}
LEAGUE_EVENT_PREFIXES = {
    'nba': ('KXNBAGAME-',),
    'wnba': ('KXWNBAGAME-',),
    'mlb': ('KXMLBGAME-',),
}
GAME_SCOPES = {
    'Games',
    'Game',
    'Head to Head',
    'Matchups',
    'Regulation Time Moneyline',
}
MLB_TEAM_ABBREVIATIONS = {
    108: 'LAA',
    109: 'AZ',
    110: 'BAL',
    111: 'BOS',
    112: 'CHC',
    113: 'CIN',
    114: 'CLE',
    115: 'COL',
    116: 'DET',
    117: 'HOU',
    118: 'KC',
    119: 'LAD',
    120: 'WSH',
    121: 'NYM',
    133: 'ATH',
    134: 'PIT',
    135: 'SD',
    136: 'SEA',
    137: 'SF',
    138: 'STL',
    139: 'TB',
    140: 'TEX',
    141: 'TOR',
    142: 'MIN',
    143: 'PHI',
    144: 'ATL',
    145: 'CWS',
    146: 'MIA',
    147: 'NYY',
    158: 'MIL',
}
WNBA_TEAM_ABBREVIATIONS = {
    1611661313: 'NY',
    1611661317: 'LV',
    1611661319: 'SEA',
    1611661320: 'LA',
    1611661321: 'PHX',
    1611661322: 'DAL',
    1611661323: 'CONN',
    1611661324: 'MIN',
    1611661325: 'IND',
    1611661326: 'CHI',
    1611661327: 'ATL',
    1611661328: 'WSH',
    1611661329: 'POR',
    1611661330: 'CLE',
    1611661331: 'GS',
    1611661332: 'TOR',
}
NBA_TEAM_ABBREVIATIONS = {
    1610612737: 'ATL',
    1610612738: 'BOS',
    1610612739: 'CLE',
    1610612740: 'NO',
    1610612741: 'CHI',
    1610612742: 'DAL',
    1610612743: 'DEN',
    1610612744: 'GS',
    1610612745: 'HOU',
    1610612746: 'LAC',
    1610612747: 'LAL',
    1610612748: 'MIA',
    1610612749: 'MIL',
    1610612750: 'MIN',
    1610612751: 'BKN',
    1610612752: 'NY',
    1610612753: 'ORL',
    1610612754: 'IND',
    1610612755: 'PHI',
    1610612756: 'PHX',
    1610612757: 'POR',
    1610612758: 'SAC',
    1610612759: 'SA',
    1610612760: 'OKC',
    1610612761: 'TOR',
    1610612762: 'UTA',
    1610612763: 'MEM',
    1610612764: 'WSH',
    1610612765: 'DET',
    1610612766: 'CHA',
}


def create_kalshi_session():
    session = requests.Session()
    retries = Retry(
        total=2,
        connect=2,
        read=1,
        status=2,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({'GET'}),
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


KALSHI_SESSION = create_kalshi_session()


def normalize_name(name):
    return re.sub(r'[^a-z0-9]+', '', str(name or '').lower())


def name_tokens(name):
    return [token for token in re.split(r'[^a-z0-9]+', str(name or '').lower()) if len(token) > 2]


def names_match(left, right):
    left_name = normalize_name(left)
    right_name = normalize_name(right)
    return bool(left_name and right_name) and (
        left_name == right_name or left_name in right_name or right_name in left_name
    )


def team_in_text(team_name, text):
    if names_match(team_name, text):
        return True
    normalized_text = normalize_name(text)
    tokens = name_tokens(team_name)
    return any(token in normalized_text for token in tokens)


def parse_dollars(value):
    if value in (None, '', '-'):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0 or parsed >= 1:
        return None
    return parsed


def implied_price_to_american(price):
    if price is None:
        return None
    decimal_odds = 1 / price
    if decimal_odds >= 2:
        return int(round((decimal_odds - 1) * 100))
    return int(round(-100 / (decimal_odds - 1)))


def buy_price(market, side):
    ask = parse_dollars(market.get(f'{side}_ask_dollars'))
    if ask is not None:
        return ask

    opposite = 'no' if side == 'yes' else 'yes'
    opposite_bid = parse_dollars(market.get(f'{opposite}_bid_dollars'))
    if opposite_bid is not None:
        return 1 - opposite_bid

    last_price = parse_dollars(market.get('last_price_dollars'))
    if last_price is None:
        return None
    return last_price if side == 'yes' else 1 - last_price


def market_text(event, market):
    return ' '.join(str(part or '') for part in (
        event.get('title'),
        event.get('sub_title'),
        market.get('title'),
        market.get('subtitle'),
        market.get('yes_sub_title'),
        market.get('no_sub_title'),
        market.get('rules_primary'),
    ))


def is_game_market_for_league(event, league):
    if str(event.get('category') or '').lower() != 'sports':
        return False
    metadata = event.get('product_metadata') or {}
    competition = metadata.get('competition')
    scope = metadata.get('competition_scope')
    allowed_competitions = LEAGUE_COMPETITIONS.get(league, set())
    if competition not in allowed_competitions:
        return False
    return scope in GAME_SCOPES


def event_relevance_score(event, home_name, away_name, league):
    if not is_game_market_for_league(event, league):
        return 0
    text = ' '.join(str(part or '') for part in (
        event.get('title'),
        event.get('sub_title'),
        event.get('category'),
    ))
    markets_text = ' '.join(market_text(event, market) for market in event.get('markets', []))
    combined = f'{text} {markets_text}'
    score = 0
    if team_in_text(home_name, combined):
        score += 2
    if team_in_text(away_name, combined):
        score += 2
    return score


def side_for_text(text, home_name, away_name):
    home_match = team_in_text(home_name, text)
    away_match = team_in_text(away_name, text)
    if home_match and not away_match:
        return 'home'
    if away_match and not home_match:
        return 'away'
    return None


def team_prices_from_market(event, market, home_name, away_name):
    prices = {}
    yes_side = side_for_text(str(market.get('yes_sub_title') or ''), home_name, away_name)
    if not yes_side:
        yes_side = side_for_text(str(market.get('subtitle') or ''), home_name, away_name)
    if not yes_side:
        yes_side = side_for_text(str(market.get('title') or ''), home_name, away_name)
    no_side = side_for_text(str(market.get('no_sub_title') or ''), home_name, away_name)

    if yes_side:
        prices[yes_side] = buy_price(market, 'yes')
    if no_side:
        prices[no_side] = buy_price(market, 'no')

    # Some Kalshi sports markets are phrased as "Will Team A beat Team B?"
    # without an explicit no subtitle. In that case NO is the opponent.
    if yes_side and not no_side:
        prices['away' if yes_side == 'home' else 'home'] = buy_price(market, 'no')

    return {side: price for side, price in prices.items() if price is not None}


def find_kalshi_event_prices(events, home_name, away_name, league):
    candidates = sorted(
        (event for event in events if event_relevance_score(event, home_name, away_name, league) >= 4),
        key=lambda event: event_relevance_score(event, home_name, away_name, league),
        reverse=True,
    )

    for event in candidates:
        prices = {}
        markets_used = {}
        for market in event.get('markets', []):
            market_prices = team_prices_from_market(event, market, home_name, away_name)
            for side, price in market_prices.items():
                if side not in prices:
                    prices[side] = price
                    markets_used[side] = market
            if 'home' in prices and 'away' in prices:
                return event, markets_used, prices
    return None, None, None


def is_kalshi_game_market(market, league):
    event_ticker = str(market.get('event_ticker') or '')
    prefixes = LEAGUE_EVENT_PREFIXES.get(league, ())
    return bool(prefixes) and event_ticker.startswith(prefixes)


def market_relevance_score(markets, home_name, away_name):
    combined = ' '.join(
        ' '.join(str(part or '') for part in (
            market.get('title'),
            market.get('subtitle'),
            market.get('yes_sub_title'),
            market.get('no_sub_title'),
            market.get('rules_primary'),
        ))
        for market in markets
    )
    score = 0
    if team_in_text(home_name, combined):
        score += 2
    if team_in_text(away_name, combined):
        score += 2
    return score


def team_yes_price_from_market(market, home_name, away_name):
    side = side_for_text(str(market.get('yes_sub_title') or ''), home_name, away_name)
    if not side:
        side = side_for_text(str(market.get('rules_primary') or ''), home_name, away_name)
    if not side:
        return None, None
    return side, buy_price(market, 'yes')


def find_kalshi_market_prices(markets, home_name, away_name, league):
    grouped = {}
    for market in markets:
        if not is_kalshi_game_market(market, league):
            continue
        grouped.setdefault(market.get('event_ticker'), []).append(market)

    candidates = sorted(
        (
            (event_ticker, event_markets)
            for event_ticker, event_markets in grouped.items()
            if market_relevance_score(event_markets, home_name, away_name) >= 4
        ),
        key=lambda item: market_relevance_score(item[1], home_name, away_name),
        reverse=True,
    )

    for event_ticker, event_markets in candidates:
        prices = {}
        markets_used = {}
        for market in event_markets:
            side, price = team_yes_price_from_market(market, home_name, away_name)
            if side and price is not None and side not in prices:
                prices[side] = price
                markets_used[side] = market
        if 'home' in prices and 'away' in prices:
            return {'event_ticker': event_ticker, 'markets': event_markets}, markets_used, prices
    return None, None, None


def parse_start_time(value):
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None


def kalshi_event_ticker_for_game(league, game):
    start_time = parse_start_time(game.get('startTime'))
    if start_time is None:
        return None
    eastern = start_time.astimezone(ZoneInfo('America/New_York'))
    if league == 'mlb':
        away_abbr = MLB_TEAM_ABBREVIATIONS.get(int(game.get('awayTeamId') or 0))
        home_abbr = MLB_TEAM_ABBREVIATIONS.get(int(game.get('homeTeamId') or 0))
        if not away_abbr or not home_abbr:
            return None
        timestamp = eastern.strftime('%y%b%d%H%M').upper()
        return f'KXMLBGAME-{timestamp}{away_abbr}{home_abbr}'
    if league == 'wnba':
        away_abbr = WNBA_TEAM_ABBREVIATIONS.get(int(game.get('awayTeamId') or 0))
        home_abbr = WNBA_TEAM_ABBREVIATIONS.get(int(game.get('homeTeamId') or 0))
        if not away_abbr or not home_abbr:
            return None
        timestamp = eastern.strftime('%y%b%d').upper()
        return f'KXWNBAGAME-{timestamp}{away_abbr}{home_abbr}'
    if league == 'nba':
        away_abbr = NBA_TEAM_ABBREVIATIONS.get(int(game.get('awayTeamId') or 0))
        home_abbr = NBA_TEAM_ABBREVIATIONS.get(int(game.get('homeTeamId') or 0))
        if not away_abbr or not home_abbr:
            return None
        timestamp = eastern.strftime('%y%b%d').upper()
        return f'KXNBAGAME-{timestamp}{away_abbr}{home_abbr}'
    return None


def kalshi_markets_for_event(event_ticker):
    if not event_ticker:
        return []
    response = KALSHI_SESSION.get(
        f'{KALSHI_API_BASE}/markets',
        params={'event_ticker': event_ticker, 'limit': 100},
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get('markets', [])


def kalshi_events():
    events = []
    cursor = ''
    params = {
        'limit': 200,
        'status': 'open',
        'with_nested_markets': 'true',
        'with_milestones': 'true',
    }
    for _ in range(KALSHI_MAX_EVENT_PAGES):
        if cursor:
            params['cursor'] = cursor
        else:
            params.pop('cursor', None)

        response = KALSHI_SESSION.get(f'{KALSHI_API_BASE}/events', params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        events.extend(payload.get('events', []))
        cursor = payload.get('cursor') or ''
        if not cursor:
            break
    return events


def kalshi_markets():
    markets = []
    cursor = ''
    params = {
        'limit': 200,
        'status': 'open',
        'category': 'Sports',
    }
    for _ in range(KALSHI_MAX_MARKET_PAGES):
        if cursor:
            params['cursor'] = cursor
        else:
            params.pop('cursor', None)

        response = KALSHI_SESSION.get(f'{KALSHI_API_BASE}/markets', params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        markets.extend(payload.get('markets', []))
        cursor = payload.get('cursor') or ''
        if not cursor:
            break
    return markets


def fetch_moneyline_odds(league, game):
    home_name = game.get('homeFullName') or game.get('home')
    away_name = game.get('awayFullName') or game.get('away')

    try:
        event_ticker = kalshi_event_ticker_for_game(league, game)
        event_markets = kalshi_markets_for_event(event_ticker)
        event, markets, prices = find_kalshi_market_prices(event_markets, home_name, away_name, league)
        if not event:
            event, markets, prices = find_kalshi_market_prices(kalshi_markets(), home_name, away_name, league)
    except requests.RequestException as error:
        raise RuntimeError('Unable to reach Kalshi market data. Please try again shortly.') from error

    if not event or not markets or not prices:
        return None

    home_price = implied_price_to_american(prices.get('home'))
    away_price = implied_price_to_american(prices.get('away'))
    if home_price is None or away_price is None:
        return None

    last_update = max(
        (
            market.get('updated_time') or event.get('last_updated_ts') or ''
            for market in markets.values()
        ),
        default='',
    ) or None

    return {
        'gameId': game.get('gameId'),
        'commenceTime': (
            event.get('strike_date')
            or event.get('expected_expiration_time')
            or next((market.get('occurrence_datetime') for market in markets.values() if market.get('occurrence_datetime')), None)
        ),
        'bookmaker': 'kalshi',
        'bookmakerTitle': 'Kalshi',
        'lastUpdate': last_update or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'eventTicker': event.get('event_ticker'),
        'away': {
            'team': game.get('away'),
            'fullName': away_name,
            'price': away_price,
            'ticker': markets['away'].get('ticker'),
            'kalshiPrice': prices.get('away'),
        },
        'home': {
            'team': game.get('home'),
            'fullName': home_name,
            'price': home_price,
            'ticker': markets['home'].get('ticker'),
            'kalshiPrice': prices.get('home'),
        },
    }
