import {useEffect, useState} from 'react'
import './App.css'
function App() {
  const [date, setDate] = useState("Date");
  const [games, setGames] = useState<{ home: string; away: string; gameId: string; }[]>([]);
  // const [players, setPlayers] = useState<{away: string[], home: string[]; } | null>();
  const [winner, setWinner] = useState();
  const [conf, setConf] = useState('%')
  const [loading, setLoading] = useState(false);


  useEffect(() => {
    fetch("http://localhost:8080/api/scoreboard").then(
      response => response.json()
    ).then(
      data => {
        setDate(data.date);
        setGames(data.teams);
      }
    )
  }, [])

  // const fetchPlayers = (id: string)  => {
  //   fetch(`http://localhost:8080/api/players/${id}`).then(
  //     res => res.json()
  //   ).then (
  //     data => setPlayers(data)
  //   );
  // };

  const fetchPredict = (id: string) => {
    setLoading(true);
    setWinner(undefined);
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
      <h1>NBA Predictor</h1>
      <h2>Today's Games ({date})</h2>
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
            ) : winner ? (
              <div className = "win"><strong>{winner}</strong> is favored to win with <strong>{conf}%</strong> confidence</div>
            ) : (<div></div>)}
            {/* {players ? (
              <div>
                <h3>away players</h3>
                <ul>{players.away.map((p,index) => <div key = {index}>{p}</div>)}</ul>
                <h3>Home Players</h3>
                <ul>{players.home.map((p,index) => <div key = {index}>{p}</div>)}</ul>
              </div>
            ) : (<p>Click a game to see players</p>)} */}
        </div>
      </div>
    </div>

  )
}

export default App
