import {useEffect, useEffectEvent, useRef, useState} from 'react';
import {GameList} from './components/GameList';
import {MatchupPanel} from './components/MatchupPanel';
import type {Game, League, LiveData, OddsData, PredictionProbabilities} from './types';
import {dateInputValue, liveDataFromGame} from './utils';
import './App.css';
import './Odds.css';

const API_BASE = 'http://localhost:8080/api';
const SCOREBOARD_POLL_MS = 30_000;
const SELECTED_GAME_POLL_MS = 15_000;

function App() {
  const [league, setLeague] = useState<League>('wnba');
  const [selectedDate, setSelectedDate] = useState(dateInputValue());
  const [date, setDate] = useState(dateInputValue());
  const [games, setGames] = useState<Game[]>([]);
  const [winner, setWinner] = useState<string>();
  const [confidence, setConfidence] = useState<number | string>('%');
  const [loading, setLoading] = useState(false);
  const [predictionMessage, setPredictionMessage] = useState('');
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [liveData, setLiveData] = useState<LiveData | null>(null);
  const [odds, setOdds] = useState<OddsData | null>(null);
  const [oddsMessage, setOddsMessage] = useState('');
  const [oddsLoading, setOddsLoading] = useState(false);
  const [oddsEnabled, setOddsEnabled] = useState(true);
  const [kalshiBetMessage, setKalshiBetMessage] = useState('');
  const [predictionProbabilities, setPredictionProbabilities] = useState<PredictionProbabilities | null>(null);
  // Refs keep asynchronous requests from updating the screen with stale data.
  const predictionRequestRef = useRef(0);
  const oddsEnabledRef = useRef(true);

  const loadOdds = async (gameId: string) => {
    // Test mode avoids calling the external market-data provider.
    if (!oddsEnabledRef.current) {
      return;
    }
    setOddsLoading(true);
    setOdds(null);
    setOddsMessage('');
    try {
      const response = await fetch(`${API_BASE}/odds/${gameId}?league=${league}&date=${selectedDate}`);
      const data = await response.json();
      if (!oddsEnabledRef.current) {
        return;
      }
      if (data.status === 'success') {
        if (data.odds) {
          setOdds(data.odds);
        } else {
          setOddsMessage('Kalshi market prices are not available.');
        }
      } else {
        setOddsMessage(data.message || 'Kalshi market prices are not available.');
      }
    } catch {
      if (oddsEnabledRef.current) {
        setOddsMessage('Kalshi market prices are not available.');
      }
    } finally {
      setOddsLoading(false);
    }
  };

  const loadScoreboard = async () => {
    const response = await fetch(`${API_BASE}/scoreboard?league=${league}&date=${selectedDate}`);
    const data = await response.json();
    setDate(data.date);
    setGames(data.teams);
    return data.teams as Game[];
  };

  const maybePlaceKalshiEdgeBet = async (gameId: string, requestId: number) => {
    try {
      const response = await fetch(`${API_BASE}/kalshi/edge-bet/${gameId}?league=${league}&date=${selectedDate}`, {method: 'POST'});
      const data = await response.json();
      if (requestId !== predictionRequestRef.current) {
        return;
      }
      if (data.status !== 'success') {
        setKalshiBetMessage(data.message || 'Kalshi edge-bet check failed.');
        return;
      }
      const result = data.result;
      const team = result?.pick?.team ? ` ${result.pick.team}` : '';
      const prefix = result?.status === 'placed'
        ? 'Kalshi order placed for'
        : result?.status === 'dry_run'
          ? 'Kalshi dry-run order prepared for'
          : result?.status === 'disabled'
            ? 'Kalshi trading disabled for'
            : 'Kalshi edge bet skipped for';
      setKalshiBetMessage(`${prefix}${team}. ${result?.reason || ''}`.trim());
      if (result?.status === 'placed') {
        void loadOdds(gameId);
      }
    } catch {
      if (requestId === predictionRequestRef.current) {
        setKalshiBetMessage('Kalshi edge-bet check failed.');
      }
    }
  };

  const fetchLive = async (gameId: string, clearOnFail = true) => {
    try {
      const response = await fetch(`${API_BASE}/live/${gameId}?league=${league}`, {cache: 'no-store'});
      const data = await response.json();
      if (data.status === 'success') {
        setLiveData(data.data);
        setPredictionProbabilities(data.data.probabilities ?? null);
        return data.data as LiveData;
      }
    } catch {
      // Fall through to the empty state below.
    }
    if (clearOnFail) {
      setLiveData(null);
    }
    return null;
  };

  const fetchPregamePrediction = async (gameId: string, requestId: number) => {
    try {
      const response = await fetch(`${API_BASE}/predict/${gameId}?league=${league}&date=${selectedDate}`);
      const data = await response.json();
      if (requestId !== predictionRequestRef.current) {
        return;
      }
      if (data.status === 'success') {
        setWinner(data.winner);
        setConfidence(data.conf);
        setPredictionProbabilities(data.probabilities ?? null);
        setPredictionMessage('');
        setLoading(false);
        void maybePlaceKalshiEdgeBet(gameId, requestId);
        return;
      }
      const message = String(data.message || '').toLowerCase();
      if (response.status === 503 || message.includes('training') || message.includes('model')) {
        setPredictionMessage(data.message || 'Training model. Retrying prediction...');
        setTimeout(() => fetchPregamePrediction(gameId, requestId), 3000);
        return;
      }
      setPredictionMessage(data.message || 'Prediction is not available for this game.');
    } catch {
      setPredictionMessage('Prediction service is not reachable.');
    }
    if (requestId === predictionRequestRef.current) {
      setLoading(false);
    }
  };

  const fetchPredict = async (gameId: string) => {
    // Every new selection gets a higher ID, so older responses can be ignored.
    const requestId = predictionRequestRef.current + 1;
    predictionRequestRef.current = requestId;
    setLoading(true);
    setWinner(undefined);
    setPredictionMessage('');
    setLiveData(null);
    setOdds(null);
    setOddsMessage('');
    setKalshiBetMessage('');
    setPredictionProbabilities(null);

    const selectedGame = games.find((game) => game.gameId === gameId);
    setSelectedGameId(gameId);
    if (oddsEnabledRef.current && selectedGame?.status === 1) {
      await loadOdds(gameId);
    }
    if (selectedGame?.isFinal) {
      setLiveData(liveDataFromGame(selectedGame));
      setLoading(false);
      return;
    }
    if (selectedGame?.isLive || selectedGame?.status === 2) {
      setLiveData(liveDataFromGame(selectedGame));
      if (oddsEnabledRef.current) {
        void loadOdds(gameId);
      }
      await fetchLive(gameId, false);
      setLoading(false);
      return;
    }
    if (selectedGame?.status === 1) {
      fetchPregamePrediction(gameId, requestId);
      return;
    }
    if (await fetchLive(gameId)) {
      setLoading(false);
      return;
    }
    fetchPregamePrediction(gameId, requestId);
  };

  const initializeScoreboard = useEffectEvent(() => {
    loadScoreboard().then(() => fetch(`${API_BASE}/train?league=${league}`));
  });

  const pollScoreboard = useEffectEvent(async (gameId: string | null) => {
      if (!gameId) {
        await loadScoreboard();
        return;
      }
      const latestGames = await loadScoreboard();
      const selectedGame = latestGames.find((game) => game.gameId === gameId);
      if (!selectedGame) {
        return;
      }
      if (selectedGame.isFinal) {
        setLiveData(liveDataFromGame(selectedGame));
      } else if (selectedGame.isLive || selectedGame.status === 2) {
        setLiveData(liveDataFromGame(selectedGame));
        await fetchLive(gameId, false);
      }
  });

  useEffect(() => {
    const initializationTimer = window.setTimeout(initializeScoreboard, 0);
    return () => window.clearTimeout(initializationTimer);
  }, [league, selectedDate]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      pollScoreboard(selectedGameId);
    }, selectedGameId ? SELECTED_GAME_POLL_MS : SCOREBOARD_POLL_MS);
    return () => window.clearInterval(interval);
  }, [selectedGameId, league, selectedDate]);

  const resetSelection = () => {
    setWinner(undefined);
    setPredictionMessage('');
    setLiveData(null);
    setSelectedGameId(null);
    setOdds(null);
    setOddsMessage('');
    setKalshiBetMessage('');
    setPredictionProbabilities(null);
    setLoading(false);
  };

  const selectLeague = (nextLeague: League) => {
    resetSelection();
    setLeague(nextLeague);
  };

  const selectDate = (nextDate: string) => {
    resetSelection();
    setSelectedDate(nextDate);
  };

  const selectedGame = games.find((game) => game.gameId === selectedGameId);

  const toggleOddsApi = () => {
    const nextEnabled = !oddsEnabledRef.current;
    oddsEnabledRef.current = nextEnabled;
    setOddsEnabled(nextEnabled);
    if (!nextEnabled) {
      setOdds(null);
      setOddsLoading(false);
      setOddsMessage('Test mode: Kalshi market-data calls are disabled.');
    } else {
      setOddsMessage('Kalshi market-data calls are enabled. Select a game to load prices.');
    }
  };

  return (
    <div>
      <header className="app-nav">
        <div className="nav-brand" aria-label="Predictor home">
          <span className="nav-mark">XD</span>
          <span className="nav-title">Predictor</span>
        </div>

        <nav className="league-toggle" aria-label="League selector">
          <button className={league === 'nba' ? 'active' : ''} onClick={() => selectLeague('nba')}>NBA</button>
          <button className={league === 'wnba' ? 'active' : ''} onClick={() => selectLeague('wnba')}>WNBA</button>
          <button className={league === 'mlb' ? 'active' : ''} onClick={() => selectLeague('mlb')}>MLB</button>
        </nav>

        <div className="nav-tools">
          <span className="nav-section-label">Scoreboard</span>
          <div className="date-control">
            <input aria-label="Scoreboard date" type="date" value={selectedDate} onChange={(event) => selectDate(event.target.value)} />
            <button onClick={() => selectDate(dateInputValue())}>Today</button>
          </div>
        </div>
      </header>

      <main className="app-shell">
        <h1 className="title">Sports Predictor</h1>
        <h2 className="title" style={{paddingBottom: '2rem'}}>Games ({date})</h2>
        <div className="app-content">
          <GameList games={games} league={league} onSelect={fetchPredict} />
          <MatchupPanel
            loading={loading}
            league={league}
            date={selectedDate}
            liveData={liveData}
            winner={winner}
            confidence={confidence}
            predictionMessage={predictionMessage}
            selectedGame={selectedGame}
            odds={odds}
            oddsLoading={oddsLoading}
            oddsEnabled={oddsEnabled}
            oddsMessage={oddsMessage}
            kalshiBetMessage={kalshiBetMessage}
            predictionProbabilities={predictionProbabilities}
            onToggleOddsApi={toggleOddsApi}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
