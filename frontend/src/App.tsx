import {useEffect, useState} from 'react'
import './App.css'

interface ScoreStats {
  gameId: string;
  gameStatusText: string;
  awayTeam: { teamCity: string; teamName: string; score: number };
  homeTeam: { teamCity: string; teamName: string; score: number };
  gameLeaders: {
    awayLeaders: { 
      personId: number;
      points: number;
      rebounds: number;
      assists: number;
     };
    homeLeaders: { 
      personId: number;
      points: number;
      rebounds: number;
      assists: number;
     };
  };
}

function App() {
  const [games, setGames] = useState<ScoreStats[]>([]);
  const [selectedNames, setSelectedNames] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [predictInfo, setPredictInfo] = useState<[string, string] | null>(null);

  const fetchScoreboard = () => {
    console.log("Refreshing scoreboard...");
    fetch("http://localhost:8080/api/scoreboard")
    .then(res => res.json())
    .then((data) => {
      setGames(data.games);
    })
    .catch(err => console.error("Error fetching scoreboard:", err));
  };

  useEffect(() => {
    fetchScoreboard();
  }, []);

  const sendData = async (game: ScoreStats) => {
    const response = await fetch("http://localhost:8080/api/game-info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id1: game.gameLeaders.awayLeaders.personId,
        id2: game.gameLeaders.homeLeaders.personId,
        team1: `${game.awayTeam.teamCity} ${game.awayTeam.teamName}`,
        team2: `${game.homeTeam.teamCity} ${game.homeTeam.teamName}`
      }),
    });
    const result = await response.json();
    setPredictInfo(result['predict-info']);
    setSelectedNames(result.names); 
    setStats(game.gameLeaders);
  };

  return (
    <div>
      <h1>NBA Predictor</h1>
      
      <div>
        <button onClick={fetchScoreboard}>Refresh Scoreboard</button>
      </div>

      <h2>Today's Games</h2>
      <ul>
        {games.map((game) => (
          <li key={game.gameId}>
            {game.awayTeam.teamName} {game.awayTeam.score} vs {game.homeTeam.teamName} {game.homeTeam.score} ({game.gameStatusText})
            <button onClick={() => sendData(game)}>View Stats & Predict</button>
          </li>
        ))}
      </ul>

      {stats && (
        <div>
          <h2>Game Analysis</h2>
          {predictInfo && (
            <div>
              <strong>Prediction:</strong> {predictInfo[0]} is favored to win ({predictInfo[1]} confidence)
            </div>
          )}
          <h3>Leader Stats</h3>
          <p>{selectedNames[0]?.full_name}: {stats.awayLeaders.points} PTS, {stats.awayLeaders.rebounds} REB, {stats.awayLeaders.assists} AST</p>
          <p>{selectedNames[1]?.full_name}: {stats.homeLeaders.points} PTS, {stats.homeLeaders.rebounds} REB, {stats.homeLeaders.assists} AST</p>
        </div>
      )}
    </div>
  )
}

export default App
