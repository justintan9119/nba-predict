import base64
import datetime
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

from odds import KALSHI_API_BASE


def bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


KALSHI_TRADING_ENABLED = bool_env('KALSHI_TRADING_ENABLED', False)
KALSHI_DRY_RUN = bool_env('KALSHI_DRY_RUN', True)
KALSHI_API_KEY_ID = os.environ.get('KALSHI_API_KEY_ID', '').strip()
KALSHI_PRIVATE_KEY_PATH = os.environ.get('KALSHI_PRIVATE_KEY_PATH', '').strip()
KALSHI_PRIVATE_KEY_PEM = os.environ.get('KALSHI_PRIVATE_KEY_PEM', '').strip()
KALSHI_BANKROLL_CENTS = max(1, int(os.environ.get('KALSHI_BANKROLL_CENTS', '9000')))
KALSHI_KELLY_FRACTION = max(0.0, float(os.environ.get('KALSHI_KELLY_FRACTION', '0.25')))
KALSHI_MAX_BANKROLL_FRACTION = max(0.0, float(os.environ.get('KALSHI_MAX_BANKROLL_FRACTION', '0.05')))
KALSHI_BET_MIN_COST_CENTS = max(1, int(os.environ.get('KALSHI_BET_MIN_COST_CENTS', '100')))
KALSHI_BET_MAX_COST_CENTS = max(1, int(os.environ.get('KALSHI_BET_MAX_COST_CENTS', '300')))
KALSHI_MIN_EDGE = float(os.environ.get('KALSHI_MIN_EDGE', '0.03'))


def load_private_key():
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
    except ImportError as error:
        raise RuntimeError('Kalshi trading requires the cryptography package in the backend environment.') from error

    if KALSHI_PRIVATE_KEY_PEM:
        key_bytes = KALSHI_PRIVATE_KEY_PEM.replace('\\n', '\n').encode('utf-8')
    elif KALSHI_PRIVATE_KEY_PATH:
        key_text = Path(KALSHI_PRIVATE_KEY_PATH).read_text(encoding='utf-8').strip()
        key_bytes = key_text.replace('\\n', '\n').encode('utf-8')
    else:
        raise RuntimeError('Set KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY_PEM before enabling Kalshi trading.')

    candidates = [key_bytes]
    key_text = key_bytes.decode('utf-8', errors='ignore').strip()
    if '-----BEGIN' not in key_text:
        compact_key = ''.join(key_text.split())
        candidates.extend([
            f'-----BEGIN PRIVATE KEY-----\n{compact_key}\n-----END PRIVATE KEY-----\n'.encode('utf-8'),
            f'-----BEGIN RSA PRIVATE KEY-----\n{compact_key}\n-----END RSA PRIVATE KEY-----\n'.encode('utf-8'),
        ])

    last_error = None
    for candidate in candidates:
        try:
            return serialization.load_pem_private_key(
                candidate,
                password=None,
                backend=default_backend(),
            )
        except ValueError as error:
            last_error = error
    raise RuntimeError('Unable to load Kalshi private key. Export it as an unencrypted PEM private key.') from last_error


def sign_request(private_key, timestamp, method, path):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    message = f'{timestamp}{method.upper()}{path.split("?")[0]}'.encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode('utf-8')


def auth_headers(method, path):
    if not KALSHI_API_KEY_ID:
        raise RuntimeError('Set KALSHI_API_KEY_ID before enabling Kalshi trading.')
    timestamp = str(int(datetime.datetime.now(datetime.UTC).timestamp() * 1000))
    private_key = load_private_key()
    sign_path = urlparse(f'{KALSHI_API_BASE}{path}').path
    return {
        'KALSHI-ACCESS-KEY': KALSHI_API_KEY_ID,
        'KALSHI-ACCESS-TIMESTAMP': timestamp,
        'KALSHI-ACCESS-SIGNATURE': sign_request(private_key, timestamp, method, sign_path),
    }


def kalshi_request(method, path, **kwargs):
    headers = kwargs.pop('headers', {})
    headers.update(auth_headers(method, path))
    if method.upper() in {'POST', 'PUT', 'PATCH'}:
        headers.setdefault('Content-Type', 'application/json')
    response = requests.request(method, f'{KALSHI_API_BASE}{path}', headers=headers, timeout=10, **kwargs)
    response.raise_for_status()
    return response.json()


def kalshi_credentials_configured():
    return bool(KALSHI_API_KEY_ID and (KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY_PEM))


def implied_probability_from_american(value):
    if value is None or value == 0:
        return None
    value = float(value)
    if value > 0:
        return 100 / (value + 100)
    return abs(value) / (abs(value) + 100)


def decimal_odds_from_american(value):
    value = float(value)
    return 1 + value / 100 if value > 0 else 1 + 100 / abs(value)


def model_pick(probabilities):
    if not probabilities:
        return None
    home = float(probabilities.get('home', 0.5))
    away = float(probabilities.get('away', 0.5))
    return 'home' if home >= away else 'away'


def evaluate_pick(odds, probabilities):
    side = model_pick(probabilities)
    if not side or not odds or side not in odds:
        return None

    raw_home = implied_probability_from_american(odds['home'].get('price'))
    raw_away = implied_probability_from_american(odds['away'].get('price'))
    if raw_home is None or raw_away is None or raw_home + raw_away <= 0:
        return None

    fair = {
        'home': raw_home / (raw_home + raw_away),
        'away': raw_away / (raw_home + raw_away),
    }
    model_probability = float(probabilities[side])
    price = odds[side].get('price')
    kalshi_price = odds[side].get('kalshiPrice')
    edge = model_probability - fair[side]
    expected_value = (model_probability * decimal_odds_from_american(price)) - 1
    return {
        'side': side,
        'ticker': odds[side].get('ticker'),
        'team': odds[side].get('fullName') or odds[side].get('team'),
        'kalshiPrice': kalshi_price,
        'americanPrice': price,
        'modelProbability': model_probability,
        'fairProbability': fair[side],
        'edge': edge,
        'contractEdge': model_probability - float(kalshi_price) if kalshi_price is not None else None,
        'expectedValue': expected_value,
    }


def existing_event_activity(event_ticker, tickers):
    positions = kalshi_request(
        'GET',
        '/portfolio/positions',
        params={'event_ticker': event_ticker, 'count_filter': 'position,total_traded', 'limit': 1000},
    )
    for position in positions.get('market_positions', []):
        if position.get('ticker') in tickers:
            return {'type': 'position', 'ticker': position.get('ticker')}

    orders = kalshi_request(
        'GET',
        '/portfolio/orders',
        params={'event_ticker': event_ticker, 'status': 'resting', 'limit': 1000},
    )
    for order in orders.get('orders', []):
        if order.get('ticker') in tickers:
            return {'type': 'resting_order', 'ticker': order.get('ticker')}
    return None


def side_for_odds_ticker(odds, ticker):
    for side in ('away', 'home'):
        if odds.get(side, {}).get('ticker') == ticker:
            return side
    return None


def team_for_odds_ticker(odds, ticker):
    side = side_for_odds_ticker(odds, ticker)
    if not side:
        return None
    return odds.get(side, {}).get('fullName') or odds.get(side, {}).get('team')


def int_value(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def kalshi_user_bets_for_odds(odds):
    if not odds or not kalshi_credentials_configured():
        return []

    event_ticker = odds.get('eventTicker')
    tickers = {odds.get('home', {}).get('ticker'), odds.get('away', {}).get('ticker')}
    tickers.discard(None)
    if not event_ticker or not tickers:
        return []

    bets = []
    positions = kalshi_request(
        'GET',
        '/portfolio/positions',
        params={'event_ticker': event_ticker, 'count_filter': 'position,total_traded', 'limit': 1000},
    )
    for position in positions.get('market_positions', []):
        ticker = position.get('ticker')
        if ticker not in tickers:
            continue
        net_position = int_value(position.get('position'))
        total_traded = int_value(position.get('total_traded'))
        if net_position == 0 and total_traded == 0:
            continue
        bets.append({
            'type': 'position',
            'ticker': ticker,
            'side': side_for_odds_ticker(odds, ticker),
            'team': team_for_odds_ticker(odds, ticker),
            'contractSide': 'YES' if net_position >= 0 else 'NO',
            'contracts': abs(net_position) or total_traded,
            'netPosition': net_position,
            'totalTraded': total_traded,
            'marketExposureCents': abs(int_value(position.get('market_exposure'))),
        })

    orders = kalshi_request(
        'GET',
        '/portfolio/orders',
        params={'event_ticker': event_ticker, 'status': 'resting', 'limit': 1000},
    )
    for order in orders.get('orders', []):
        ticker = order.get('ticker')
        if ticker not in tickers:
            continue
        count = int_value(order.get('count'))
        remaining_count = int_value(order.get('remaining_count'), count)
        yes_price = int_value(order.get('yes_price'))
        no_price = int_value(order.get('no_price'))
        order_side = str(order.get('side') or 'yes').upper()
        price_cents = yes_price if order_side == 'YES' else no_price
        if price_cents <= 0 and yes_price > 0:
            price_cents = yes_price
        bets.append({
            'type': 'resting_order',
            'ticker': ticker,
            'side': side_for_odds_ticker(odds, ticker),
            'team': team_for_odds_ticker(odds, ticker),
            'contractSide': order_side,
            'action': str(order.get('action') or 'buy').upper(),
            'contracts': count,
            'remainingContracts': remaining_count,
            'priceCents': price_cents,
            'maxCostCents': remaining_count * price_cents if price_cents > 0 else None,
            'createdTime': order.get('created_time'),
        })

    return bets


def recommended_stake_cents(kalshi_price, model_probability):
    price = float(kalshi_price)
    probability = float(model_probability)
    if price <= 0 or price >= 1 or probability <= price:
        return 0

    full_kelly = (probability - price) / (1 - price)
    raw_stake = KALSHI_BANKROLL_CENTS * full_kelly * KALSHI_KELLY_FRACTION
    bankroll_cap = KALSHI_BANKROLL_CENTS * KALSHI_MAX_BANKROLL_FRACTION
    max_stake = min(KALSHI_BET_MAX_COST_CENTS, int(bankroll_cap))
    if max_stake <= 0:
        return 0
    return int(round(max(KALSHI_BET_MIN_COST_CENTS, min(raw_stake, max_stake))))


def order_count_for_price(kalshi_price, model_probability):
    price_cents = max(1, min(99, int(round(float(kalshi_price) * 100))))
    stake_cents = recommended_stake_cents(kalshi_price, model_probability)
    if stake_cents < price_cents:
        return 0, price_cents, stake_cents
    count = stake_cents // price_cents
    return max(1, count), price_cents, stake_cents


def maybe_place_edge_bet(game, odds, probabilities):
    if game.get('status') != 1:
        return {'status': 'skipped', 'reason': 'Bets are only considered before the game starts.'}
    if not odds:
        return {'status': 'skipped', 'reason': 'Kalshi prices are unavailable.'}

    pick = evaluate_pick(odds, probabilities)
    if not pick or not pick.get('ticker') or pick.get('kalshiPrice') is None:
        return {'status': 'skipped', 'reason': 'No valid model-side Kalshi market was available.'}
    if pick['contractEdge'] is None or pick['contractEdge'] < KALSHI_MIN_EDGE or pick['expectedValue'] <= 0:
        return {'status': 'skipped', 'reason': 'Model side does not meet the configured edge and EV thresholds.', 'pick': pick}

    event_ticker = odds.get('eventTicker')
    tickers = {odds['home'].get('ticker'), odds['away'].get('ticker')}
    tickers.discard(None)

    if not KALSHI_TRADING_ENABLED:
        return {'status': 'disabled', 'reason': 'Set KALSHI_TRADING_ENABLED=1 to allow live Kalshi order placement.', 'pick': pick}

    count, price_cents, stake_cents = order_count_for_price(pick['kalshiPrice'], pick['modelProbability'])
    if count <= 0:
        return {'status': 'skipped', 'reason': 'Kelly sizing produced no valid contract count.', 'pick': pick}
    order = {
        'ticker': pick['ticker'],
        'side': 'yes',
        'action': 'buy',
        'count': count,
        'yes_price': price_cents,
        'time_in_force': 'immediate_or_cancel',
        'client_order_id': f'edge-{game.get("gameId")}-{pick["side"]}-{uuid.uuid4().hex[:12]}',
        'buy_max_cost': count * price_cents,
        'recommended_stake_cents': stake_cents,
    }

    if KALSHI_DRY_RUN:
        return {'status': 'dry_run', 'reason': 'Set KALSHI_DRY_RUN=0 to submit real orders.', 'pick': pick, 'order': order}

    existing = existing_event_activity(event_ticker, tickers)
    if existing:
        return {'status': 'skipped', 'reason': 'Existing Kalshi position or resting order found for this game.', 'existing': existing, 'pick': pick}

    result = kalshi_request('POST', '/portfolio/orders', json=order)
    return {'status': 'placed', 'pick': pick, 'order': result.get('order', result)}
