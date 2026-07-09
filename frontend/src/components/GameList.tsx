import type {Game, League} from '../types';
import {formatClock} from '../utils';

const hideBrokenLogo = (event: React.SyntheticEvent<HTMLImageElement>) => {
  event.currentTarget.style.display = 'none';
};

const formatStartTime = (startTime?: string) => {
  if (!startTime) {
    return '';
  }
  const parsed = new Date(startTime);
  if (Number.isNaN(parsed.getTime())) {
    return startTime;
  }
  return parsed.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
};

type GameListProps = {
  games: Game[];
  league: League;
  onSelect: (gameId: string) => void;
};

function gameStatusClass(game: Game) {
  if (game.isLive || game.status === 2) {
    return 'live';
  }
  if (game.isFinal || game.status === 3) {
    return 'final';
  }
  return 'pregame';
}

export function GameList({games, league, onSelect}: GameListProps) {
  if (games.length === 0) {
    return <div className="game-slider-empty">There are no games today</div>;
  }

  return (
    <div className="game-slider" aria-label="Games">
      {games.map((game) => (
        <div className={`gameList ${gameStatusClass(game)}`} key={game.gameId}>
          <div className="scoreboard-status">
            {game.isFinal ? 'FINAL' : game.isLive || game.status === 2 ? 'LIVE' : game.statusText || 'PRE-GAME'}
          </div>
          <div className="scoreboard-matchup">
            <span className="scoreboard-team">{game.awayLogo && <img src={game.awayLogo} alt="" onError={hideBrokenLogo} />}{game.away}</span>
            <strong>{game.awayScore ?? 0}</strong>
          </div>
          <div className="scoreboard-matchup">
            <span className="scoreboard-team">{game.homeLogo && <img src={game.homeLogo} alt="" onError={hideBrokenLogo} />}{game.home}</span>
            <strong>{game.homeScore ?? 0}</strong>
          </div>
          <div className="scoreboard-clock">
            {game.isFinal ? 'Game Ended' : game.isLive || game.status === 2
              ? league === 'mlb' ? game.clock || 'In progress' : `Q${game.period ?? 0} ${formatClock(game.clock)}`
              : formatStartTime(game.startTime) || game.statusText || 'Not started'}
          </div>
          <button className="stats" onClick={() => onSelect(game.gameId)}>Open Stats</button>
        </div>
      ))}
    </div>
  );
}
