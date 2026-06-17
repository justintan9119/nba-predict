import {useEffect, useState} from 'react'
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

type LiveData = {
  winner: string;
  confidence?: number;
  home: string;
  away: string;
  score: string;
  clock: string;
  period: number;
  statusText?: string;
  isFinal?: boolean;
};

type Game = {
  home: string;
  away: string;
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
    score: `${awayScore} - ${homeScore}`,
    clock: game.isFinal ? 'FINAL' : formatClock(game.clock || game.statusText),
    period: game.period ?? 0,
    statusText: game.statusText
  };
};

function App() {
  const [league, setLeague] = useState<League>('wnba');
  const [date, setDate] = useState("Date");
  const [games, setGames] = useState<Game[]>([]);
  // const [players, setPlayers] = useState<{away: string[], home: string[]; } | null>();
  const [winner, setWinner] = useState();
  const [conf, setConf] = useState('%')
  const [loading, setLoading] = useState(false);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [liveData, setLiveData] = useState<LiveData | null>(null);


  useEffect(() => {
    setWinner(undefined);
    setLiveData(null);
    setSelectedGameId(null);
    setLoading(false);

    fetch(`${API_BASE}/scoreboard?league=${league}`).then(
      response => response.json()
    ).then(
      data => {
        setDate(data.date);
        setGames(data.teams);
        
        fetch(`${API_BASE}/train?league=${league}`);
      }
    )
  }, [league])

  // Poll for live stats if we have a game selected
  useEffect(() => {
    let interval: number;
    if (selectedGameId) {
      interval = setInterval(() => {
        fetchLive(selectedGameId, false);
      }, 5000); 
    }
    return () => clearInterval(interval);
  }, [selectedGameId, league]);

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

  const fetchPredict = async (id: string) => {
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

    fetch(`${API_BASE}/predict/${id}?league=${league}`).then(
      res => res.json()
    ).then(
      data => {
        if (data.status !== 'success') {
          setLoading(false);
          return;
        }

        setWinner(data.winner)
        setConf(data.conf)
        setLoading(false);
      }
    ).catch(() => setLoading(false));
  }

  return (

    <div>
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
      <h1 className = "title">{league.toUpperCase()} Predictor</h1>
      <h2 className = "title" style={{ paddingBottom: '2rem' }}>Today's Games ({date})</h2>
      <div className = "left-side">
        {games && games.length > 0 ? (
          games.map((game) => (
            <div className = "gameList" key={game.gameId}>
              <strong>{game.home} vs {game.away}</strong> 
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
                  <span className="team-name" style={{ color: TEAM_COLORS[liveData.away] || '#f8ef39', marginRight: '1rem' }}>{liveData.away}</span>
                  <span className="live-score">{liveData.score}</span>
                  <span className="team-name" style={{ color: TEAM_COLORS[liveData.home] || '#f8ef39', marginLeft: '1rem' }}>{liveData.home}</span>
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
                      <span className="team-name" style={{ color: TEAM_COLORS[game?.away || ''] || '#f8ef39', marginRight: '1rem' }}>{game?.away}</span>
                      <span className="live-score">VS</span>
                      <span className="team-name" style={{ color: TEAM_COLORS[game?.home || ''] || '#f8ef39', marginLeft: '1rem' }}>{game?.home}</span>
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
