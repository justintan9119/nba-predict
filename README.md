# NBA Predictor

A simplified full-stack application for predicting NBA game winners using team stats and player health data.

## Project Structure

### Backend (`/backend`)
- `app.py`: The Flask server that provides API endpoints for the frontend.
- `logic.py`: The core engine that trains the machine learning model and calculates winner predictions based on 

### Frontend (`/frontend`)
- A React application (built with Vite) that displays today's games and allows you to predict winners with a single click.

## Code Guide

- `frontend/src/App.tsx`: Connects the page to backend API calls and stores page-level state.
- `frontend/src/components/`: Small UI pieces such as the game list, prediction panel, and live player stats.
- `frontend/src/Odds.css`: Shared styling for team logos, player stats, and the moneyline odds display.
- `frontend/src/types.ts`: Shared TypeScript shapes for games, odds, and predictions.
- `backend/app.py`: Flask routes that the frontend calls.
- `backend/predict.py`: Data preparation, model training, cache loading, and prediction calculations.
- `backend/mlb_predict.py`: MLB schedule, sklearn model training, live score, and advanced-metric winner prediction through `python-mlb-statsapi`.
- `backend/odds.py`: Kalshi market-price integration.

## How to Run

1. **Update Data (Optional):**
   ```bash
   cd backend
   python refresh_data.py
   ```
2. **Start Backend:**
   ```bash
   pip install python-mlb-statsapi
   ```
   ```bash
   python app.py
   ```
3. **Start Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

## Probabilities And Odds

`/api/predict/<game_id>` returns calibrated, injury-adjusted `probabilities.home` and `probabilities.away`, plus the underlying pre-injury `modelProbabilities` values. `/api/model-diagnostics?league=wnba` exposes the held-out log loss, Brier score, and calibration bins used to inspect whether predicted probability bands match actual win rates.

For matchups, `/api/odds/<game_id>` returns American-style prices converted from Kalshi YES/NO contract prices when a matching sports event is available. The frontend displays those market prices alongside the model projection.

Kalshi order placement is disabled by default. To test the model-side edge order flow without placing real orders:

```env
KALSHI_TRADING_ENABLED=1
KALSHI_DRY_RUN=1
KALSHI_BANKROLL_CENTS=9000
KALSHI_KELLY_FRACTION=0.25
KALSHI_MAX_BANKROLL_FRACTION=0.05
KALSHI_BET_MIN_COST_CENTS=100
KALSHI_BET_MAX_COST_CENTS=300
KALSHI_MIN_EDGE=0.03
```

Stake size uses quarter Kelly for a YES contract: `bankroll * ((model_probability - kalshi_price) / (1 - kalshi_price)) * 0.25`, then applies a $1 minimum and the lower of the hard cap or 5% of bankroll. To submit real orders, install `cryptography` in the backend environment, set `KALSHI_API_KEY_ID` plus either `KALSHI_PRIVATE_KEY_PATH` or `KALSHI_PRIVATE_KEY_PEM`, then set `KALSHI_DRY_RUN=0`. The app only considers orders before game start, only for the model-projected winner, only when Kalshi contract edge (`model_probability - kalshi_price`) and EV are positive, and it checks existing Kalshi positions/resting orders for that game before sending a real order.

## Model Startup

Training runs once per league in a background worker. The trained model, feature list, validation metrics, and team snapshots are cached in `backend/model_cache/` for the current day, so a restart can load them without re-downloading league logs. The first uncached run uses five seasons by default; set `MODEL_TRAINING_SEASONS` to a lower value for faster experiments or a higher value for more historical coverage.
