import {useEffect, useRef, useState} from 'react'
import './App.css'

const API_BASE = 'http://localhost:8080/api';

const TEAM_COLORS: Record<string, string> = {
  'Knicks': '#F58426',
  'Spurs': '#C4CED4',
  'Lakers': '#FDB927',
  'Celtics': '#007A33',
  'Warriors': '#FFC72C',
  'Bulls': '#CE1141',
  'Suns': '#E56020',
  'Heat': '#98002E',
  '76ers': '#006BB6',
  'Nets': '#FFFFFF',
  'Bucks': '#EEE1AF',
  'Cavaliers': '#860038',
  'Mavericks': '#00538C',
  'Nuggets': '#FEC524',
  'Pistons': '#ED174C',
  'Rockets': '#CE1141',
  'Pacers': '#FDBB30',
  'Clippers': '#C8102E',
  'Grizzlies': '#5D76A9',
  'Timberwolves': '#236192',
  'Pelicans': '#C8102E',
  'Thunder': '#007AC1',
  'Magic': '#0077C0',
  'Kings': '#5A2D81',
  'Trail Blazers': '#E03A3E',
  'Jazz': '#002B5C',
  'Hawks': '#E03A3E',
  'Hornets': '#00788C',
  'Raptors': '#CE1141',
  'Wizards': '#002B5C',
};

type League = 'nba' | 'wnba';

const dateInputValue = (date = new Date()) => {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return localDate.toISOString().slice(0, 10);
};


type LiveData = {
  winner: string;
  confidence?: number;
  home: string;
  away: string;
  homeFullName?: string;
  awayFullName?: string;
  score: string;
  clock: string;
  period: number;
  statusText?: string;
  isFinal?: boolean;
};

type Game = {
  home: string;
  away: string;
  homeFullName?: string;
  awayFullName?: string;
  gameId: string;
  awayScore?: number;
  homeScore?: number;
  status?: number;
  statusText?: string;
  clock?: string;
  period?: number;
  isLive?: boolean;
  isFinal?: boolean;
};

const formatClock = (clock?: string) => {
  const match = clock?.match(/PT(\d+)M(\d+(?:\.\d+)?)S?/);
  if (!match) {
    return clock || 'LIVE';
  }

  return `${Number(match[1])}:${Math.floor(Number(match[2])).toString().padStart(2, '0')}`;
};



const liveDataFromGame = (game: Game): LiveData => {
  const homeScore = game.homeScore ?? 0;
  const awayScore = game.awayScore ?? 0;
  const winner = homeScore === awayScore ? '' : homeScore > awayScore ? game.home : game.away;

  return {
    winner,
    isFinal: game.isFinal,
    home: game.home,
    away: game.away,
    homeFullName: game.homeFullName,
    awayFullName: game.awayFullName,
    score: `${awayScore} - ${homeScore}`,
    clock: game.isFinal ? 'FINAL' : formatClock(game.clock || game.statusText),
    period: game.period ?? 0,
    statusText: game.statusText
  };
};

function App() {
  const [league, setLeague] = useState<League>('wnba');
  const [selectedDate, setSelectedDate] = useState(dateInputValue());
  const [date, setDate] = useState(dateInputValue());
  const [games, setGames] = useState<Game[]>([]);
  // const [players, setPlayers] = useState<{away: string[], home: string[]; } | null>();
  const [winner, setWinner] = useState();
  const [conf, setConf] = useState('%')
  const [loading, setLoading] = useState(false);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [liveData, setLiveData] = useState<LiveData | null>(null);
  const predictionRequestRef = useRef(0);


  const loadScoreboard = async () => {
    const response = await fetch(`${API_BASE}/scoreboard?league=${league}&date=${selectedDate}`);
    const data = await response.json();
    setDate(data.date);
    setGames(data.teams);
    return data.teams as Game[];
  }

  useEffect(() => {
    setWinner(undefined);
    setLiveData(null);
    setSelectedGameId(null);
    setLoading(false);

    loadScoreboard().then(() => {
      fetch(`${API_BASE}/train?league=${league}`);
    });
  }, [league, selectedDate])

  useEffect(() => {
    const interval = setInterval(() => {
      loadScoreboard();
    }, 5000);

    return () => clearInterval(interval);
  }, [league, selectedDate]);

  // Poll both scoreboard and live projection for selected games.
  useEffect(() => {
    let interval: number;
    if (selectedGameId) {
      interval = setInterval(async () => {
        const latestGames = await loadScoreboard();
        const selectedGame = latestGames.find(game => game.gameId === selectedGameId);
        if (selectedGame?.isLive || selectedGame?.isFinal || selectedGame?.status === 2) {
          setLiveData(liveDataFromGame(selectedGame));
        }
        await fetchLive(selectedGameId, false);
      }, 5000); 
    }
    return () => clearInterval(interval);
  }, [selectedGameId, league, selectedDate]);

  const fetchLive = async (id: string, clearOnFail = true) => {
    try {
      const res = await fetch(`${API_BASE}/live/${id}?league=${league}`);
      const json = await res.json();
      if (json.status === 'success') {
        setLiveData(json.data);
        return json.data as LiveData;
      }
    } catch {
      // Fall through to the shared empty state below.
    }

    if (clearOnFail) {
      setLiveData(null);
    }
    return null;
  }

  const fetchPregamePrediction = async (id: string, requestId: number) => {
    try {
      const res = await fetch(`${API_BASE}/predict/${id}?league=${league}`);
      const data = await res.json();

      if (requestId !== predictionRequestRef.current) {
        return;
      }

      if (data.status === 'success') {
        setWinner(data.winner);
        setConf(data.conf);
        setLoading(false);
        return;
      }

      const message = String(data.message || '').toLowerCase();
      if (res.status === 503 || message.includes('training') || message.includes('model')) {
        setTimeout(() => fetchPregamePrediction(id, requestId), 3000);
        return;
      }
    } catch {
      // Network failures end the loading state; model warm-up responses retry above.
    }

    if (requestId === predictionRequestRef.current) {
      setLoading(false);
    }
  }

  const fetchPredict = async (id: string) => {
    const requestId = predictionRequestRef.current + 1;
    predictionRequestRef.current = requestId;
    setLoading(true);
    setWinner(undefined);
    setLiveData(null);

    const selectedGame = games.find(game => game.gameId === id);
    if (selectedGame?.isFinal) {
      setSelectedGameId(null);
      setLiveData(liveDataFromGame(selectedGame));
      setLoading(false);
      return;
    }

    setSelectedGameId(id);

    if (selectedGame?.isLive || selectedGame?.status === 2) {
      setLiveData(liveDataFromGame(selectedGame));
      await fetchLive(id, false);
      setLoading(false);
      return;
    }

    const currentLiveData = await fetchLive(id);
    if (currentLiveData) {
      setLoading(false);
      return;
    }

    fetchPregamePrediction(id, requestId);
  }

  return (

    <div>
      <div className="top-controls">
        <div className="league-toggle" aria-label="League selector">
          <button
            className={league === 'nba' ? 'active' : ''}
            onClick={() => setLeague('nba')}
          >
            NBA
          </button>
          <button
            className={league === 'wnba' ? 'active' : ''}
            onClick={() => setLeague('wnba')}
          >
            WNBA
          </button>
        </div>
        <div className="date-control">
          <input
            aria-label="Scoreboard date"
            type="date"
            value={selectedDate}
            onChange={(event) => setSelectedDate(event.target.value)}
          />
          <button onClick={() => setSelectedDate(dateInputValue())}>Today</button>
        </div>
      </div>
      <h1 className = "title">{league.toUpperCase()} Predictor</h1>
      <h2 className = "title" style={{ paddingBottom: '2rem' }}>Games ({date})</h2>
      <div className = "left-side">
        {games && games.length > 0 ? (
          games.map((game) => (
            <div className = "gameList" key={game.gameId}>
              <div className="scoreboard-status">
                {game.isFinal ? 'FINAL' : game.isLive || game.status === 2 ? 'LIVE' : game.statusText || 'PRE-GAME'}
              </div>
              <div className="scoreboard-matchup">
                <span>{game.away}</span>
                <strong>{game.awayScore ?? 0}</strong>
              </div>
              <div className="scoreboard-matchup">
                <span>{game.home}</span>
                <strong>{game.homeScore ?? 0}</strong>
              </div>
              <div className="scoreboard-clock">
                {game.isFinal ? 'Game Ended' : game.isLive || game.status === 2 ? `Q${game.period ?? 0} ${formatClock(game.clock)}` : game.statusText || 'Not started'}
              </div>
              { <button className = "stats" onClick = {() => fetchPredict(game.gameId)}>Open Stats</button> }
            </div>
          ))) : (<div>There are no games today</div>)}
      </div>
      <div className = "right-side">
        <div>
            {loading ? (
              <div className="loader">Analyzing matchup data...</div>
            ) : liveData ? (
              <div className="live-container" style={{ borderColor: liveData.isFinal ? '#555' : '#ff4444' }}>
                <div className="live-badge" style={{ backgroundColor: liveData.isFinal ? '#555' : '#ff4444' }}>
                  {liveData.isFinal ? 'FINAL' : 'LIVE'}
                </div>
                <div className="live-score-container">
                  <span className="team-name" style={{ color: TEAM_COLORS[liveData.away] || '#f8ef39' }}>{liveData.awayFullName || liveData.away}</span>
                  <span className="live-score">{liveData.score}</span>
                  <span className="team-name" style={{ color: TEAM_COLORS[liveData.home] || '#f8ef39' }}>{liveData.homeFullName || liveData.home}</span>
                </div>
                <div className="live-clock">{liveData.isFinal ? 'Game Ended' : `Quarter ${liveData.period} - ${formatClock(liveData.clock)}`}</div>
                <div className="win" style={{ marginTop: '1rem', color: liveData.isFinal ? '#fff' : '#646cff' }}>
                  {liveData.isFinal ? (
                    <></>
                  ) : liveData.confidence === undefined ? (
                    <>Live score is in progress</>
                  ) : (
                    <>Live Projection: <strong>{liveData.winner}</strong> is favored (<strong>{liveData.confidence}%</strong>)</>
                  )}
                </div>
              </div>
            ) : winner ? (() => {
                const game = games.find(g => g.gameId === selectedGameId);
                return (
                  <div className="live-container" style={{ borderColor: '#646cff' }}>
                    <div className="live-badge" style={{ backgroundColor: '#646cff' }}>PRE-GAME</div>
                    <div className="live-score-container">
                      <span className="team-name" style={{ color: TEAM_COLORS[game?.away || ''] || '#f8ef39' }}>{game?.awayFullName || game?.away}</span>
                      <span className="live-score">VS</span>
                      <span className="team-name" style={{ color: TEAM_COLORS[game?.home || ''] || '#f8ef39' }}>{game?.homeFullName || game?.home}</span>
                    </div>
                    <div className="live-clock">Matchup Analysis</div>
                    <div className="win" style={{ marginTop: '1rem' }}>
                      Projection: <strong>{winner}</strong> is favored to win (<strong>{conf}%</strong>)
                    </div>
                  </div>
                );
              })() : (<div>Select a game to see prediction</div>)}
        </div>
      </div>
    </div>

  )
}

export default App
