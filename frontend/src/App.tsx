import {useEffect, useState} from 'react'
import './App.css'


function App() {
  const [date, setDate] = useState("Date");
  const [games, setGames] = useState([["Team", "Team"]]);
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

  return (

    <body>
      <h1>NBA Predictor</h1>
      <h2>Today's Games ({date})</h2>
      {games.length > 0 ? (
        games.map((game, index) => (
          <div key ={index} className = "gameList">
            <strong>{game[0]} vs {game[1]}</strong> 
            <br></br>
            <button className = "stats">Open Stats</button>
          </div>
        ))
      ) : (
        <div>There are no games today</div>
      )
      }

    </body>
  )
}

export default App
