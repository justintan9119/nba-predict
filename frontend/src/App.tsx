import {useEffect, useState} from 'react'
import './App.css'

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

function App() {
  const [date, setDate] = useState("Date");
  const [games, setGames] = useState<{ home: string; away: string; gameId: string; }[]>([]);
  // const [players, setPlayers] = useState<{away: string[], home: string[]; } | null>();
  const [winner, setWinner] = useState();
  const [conf, setConf] = useState('%')
  const [loading, setLoading] = useState(false);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [liveData, setLiveData] = useState<{ winner: string; confidence: number; home: string; away: string; score: string; clock: string; period: number; isFinal?: boolean; } | null>(null);


  useEffect(() => {
    fetch("http://localhost:8080/api/scoreboard").then(
      response => response.json()
    ).then(
      data => {
        setDate(data.date);
        setGames(data.teams);
        
        fetch("http://localhost:8080/api/train");
      }
    )
  }, [])

  // Poll for live stats if we have a game selected
  useEffect(() => {
    let interval: number;
    if (selectedGameId) {
      interval = setInterval(() => {
        fetchLive(selectedGameId);
      }, 5000); 
    }
    return () => clearInterval(interval);
  }, [selectedGameId]);

  const fetchLive = (id: string) => {
    fetch(`http://localhost:8080/api/live/${id}`).then(
      res => res.json()
    ).then(
      json => {
        if (json.status === 'success') {
          setLiveData(json.data);
        } else {
          setLiveData(null);
        }
      }
    ).catch(() => setLiveData(null));
  }

  const fetchPredict = (id: string) => {
    setLoading(true);
    setWinner(undefined);
    setLiveData(null);
    setSelectedGameId(id);
    fetchLive(id);

    fetch(`http://localhost:8080/api/predict/${id}`).then(
      res => res.json()
    ).then(
      data => {
        setWinner(data.winner)
        setConf(data.conf)
        setLoading(false);
      }
    ).catch(() => setLoading(false));
  }

  return (

    <div>
      <h1 className = "title">NBA Predictor</h1>
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
                <div className="live-clock">{liveData.isFinal ? 'Game Ended' : `Quarter ${liveData.period} - ${liveData.clock}`}</div>
                <div className="win" style={{ marginTop: '1rem', color: liveData.isFinal ? '#fff' : '#646cff' }}>
                  {liveData.isFinal ? (
                    <></>
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
