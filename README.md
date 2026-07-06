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
- `backend/odds.py`: Moneyline odds API integration.

## How to Run

1. **Update Data (Optional):**
   ```bash
   cd backend
   python refresh_data.py
   ```
2. **Start Backend:**
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

For pre-game matchups, `/api/odds/<game_id>` returns American moneylines when a matching event is available from The Odds API. The frontend displays those moneylines alongside the model projection.

## Model Startup

Training runs once per league in a background worker. The trained model, feature list, validation metrics, and team snapshots are cached in `backend/model_cache/` for the current day, so a restart can load them without re-downloading league logs. The first uncached run uses five seasons by default; set `MODEL_TRAINING_SEASONS` to a lower value for faster experiments or a higher value for more historical coverage.
