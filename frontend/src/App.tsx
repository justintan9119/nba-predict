import {useEffect, useState} from 'react'
import './App.css'

// interface Player { 
//   id: number; 
//   full_name: string; 
// }
interface ScoreStats {
  gameId: string;
  gameStatusText: string;
  awayTeam: { teamName: string; score: number };
  homeTeam: { teamName: string; score: number };
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
  // const [players, setPlayers] = useState<Player[]>([]);
  const [games, setGames] = useState<ScoreStats[]>([]);
  const [selectedNames, setSelectedNames] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null); // New state to hold the stats

  useEffect(() => {
    // fetch("http://localhost:8080/api/players")
    // .then(res => res.json())
    // .then(data => setPlayers(data.players));
    
    fetch("http://localhost:8080/api/scoreboard")
    .then(res => res.json())
    .then((data) => {
      setGames(data.games);
      console.log(data.games);
    });
  }, []);

  const sendData = async (game: ScoreStats) => {
    const response = await fetch("http://localhost:8080/api/key-player-id", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id1: game.gameLeaders.awayLeaders.personId,
        id2: game.gameLeaders.homeLeaders.personId
      }),
    });
    const result = await response.json();
    setSelectedNames(result.names); 
    setStats(game.gameLeaders); // Save the stats from the game we clicked
  };


  return (
    <div>
      <h1>NBA Predictor</h1>
      <h2>Today's Games</h2>
      <ul>
        {games.map((game) => (
          <li key={game.gameId}>
            {game.awayTeam.teamName} {game.awayTeam.score} vs {game.homeTeam.teamName} {game.homeTeam.score} ({game.gameStatusText})
            <button onClick={() => sendData(game)}>View Stats</button>
          </li>
        ))}
      </ul>

      {/* Show stats if we have them */}
      {stats && (
        <div>
          <h2>Leader Stats</h2>
          <p>{selectedNames[0]?.full_name}: {stats.awayLeaders.points} PTS, {stats.awayLeaders.rebounds} REB, {stats.awayLeaders.assists} AST</p>
          <p>{selectedNames[1]?.full_name}: {stats.homeLeaders.points} PTS, {stats.homeLeaders.rebounds} REB, {stats.homeLeaders.assists} AST</p>
        </div>
      )}
    </div>
  )
}

export default App
