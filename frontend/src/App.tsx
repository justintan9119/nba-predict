import {useEffect, useEffectEvent, useRef, useState} from 'react';
import {GameList} from './components/GameList';
import {MatchupPanel} from './components/MatchupPanel';
import type {Game, League, LiveData, OddsData} from './types';
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
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [liveData, setLiveData] = useState<LiveData | null>(null);
  const [odds, setOdds] = useState<OddsData | null>(null);
  const [oddsMessage, setOddsMessage] = useState('');
  const [oddsLoading, setOddsLoading] = useState(false);
  const [oddsEnabled, setOddsEnabled] = useState(true);
  // Refs keep asynchronous requests from updating the screen with stale data.
  const predictionRequestRef = useRef(0);
  const oddsEnabledRef = useRef(true);

  const loadOdds = async (gameId: string) => {
    // Test mode avoids spending odds-provider API credits.
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
        setOdds(data.odds);
      } else {
        setOddsMessage(data.message || 'Moneyline odds are not available.');
      }
    } catch {
      if (oddsEnabledRef.current) {
        setOddsMessage('Moneyline odds are not available.');
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

  const fetchLive = async (gameId: string, clearOnFail = true) => {
    try {
      const response = await fetch(`${API_BASE}/live/${gameId}?league=${league}`);
      const data = await response.json();
      if (data.status === 'success') {
        setLiveData(data.data);
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
      const response = await fetch(`${API_BASE}/predict/${gameId}?league=${league}`);
      const data = await response.json();
      if (requestId !== predictionRequestRef.current) {
        return;
      }
      if (data.status === 'success') {
        setWinner(data.winner);
        setConfidence(data.conf);
        setLoading(false);
        return;
      }
      const message = String(data.message || '').toLowerCase();
      if (response.status === 503 || message.includes('training') || message.includes('model')) {
        setTimeout(() => fetchPregamePrediction(gameId, requestId), 3000);
        return;
      }
    } catch {
      // Network failures end loading below.
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
    setLiveData(null);
    setOdds(null);
    setOddsMessage('');

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
      await fetchLive(gameId, false);
      setLoading(false);
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
    setLiveData(null);
    setSelectedGameId(null);
    setOdds(null);
    setOddsMessage('');
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
      setOddsMessage('Test mode: Odds API calls are disabled.');
    } else {
      setOddsMessage('Odds API calls are enabled. Select a game to load its moneylines.');
    }
  };

  return (
    <div>
      <div className="top-controls">
        <div className="league-toggle" aria-label="League selector">
          <button className={league === 'nba' ? 'active' : ''} onClick={() => selectLeague('nba')}>NBA</button>
          <button className={league === 'wnba' ? 'active' : ''} onClick={() => selectLeague('wnba')}>WNBA</button>
        </div>
        <div className="date-control">
          <input aria-label="Scoreboard date" type="date" value={selectedDate} onChange={(event) => selectDate(event.target.value)} />
          <button onClick={() => selectDate(dateInputValue())}>Today</button>
        </div>
      </div>
      <h1 className="title">{league.toUpperCase()} Predictor</h1>
      <h2 className="title" style={{paddingBottom: '2rem'}}>Games ({date})</h2>
      <GameList games={games} onSelect={fetchPredict} />
      <MatchupPanel
        loading={loading}
        league={league}
        date={selectedDate}
        liveData={liveData}
        winner={winner}
        confidence={confidence}
        selectedGame={selectedGame}
        odds={odds}
        oddsLoading={oddsLoading}
        oddsEnabled={oddsEnabled}
        oddsMessage={oddsMessage}
        onToggleOddsApi={toggleOddsApi}
      />
    </div>
  );
}

export default App;
