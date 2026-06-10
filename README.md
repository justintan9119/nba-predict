# NBA Predictor

A simplified full-stack application for predicting NBA game winners using team stats and player health data.

## Project Structure

### Backend (`/backend`)
- `app.py`: The Flask server that provides API endpoints for the frontend.
- `logic.py`: The core engine that trains the machine learning model and calculates winner predictions based on 

### Frontend (`/frontend`)
- A React application (built with Vite) that displays today's games and allows you to predict winners with a single click.

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
