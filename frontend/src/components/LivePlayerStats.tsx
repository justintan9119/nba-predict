import {useEffect, useState} from 'react';
import type {Game, League, LiveGameStats} from '../types';

const API_BASE = 'http://localhost:8080/api';
const REFRESH_INTERVAL_MS = 15_000;

type LivePlayerStatsProps = {
  game?: Game;
  league: League;
  date: string;
};

const hideBrokenLogo = (event: React.SyntheticEvent<HTMLImageElement>) => {
  event.currentTarget.style.display = 'none';
};

function BasketballPlayerRows({players}: {players: LiveGameStats['away']}) {
  return (
    <>
      <div className="player-stat-header">
        <span>Player</span><span>PTS</span><span>REB</span><span>AST</span><span>FG</span><span>3PT</span><span>STL</span><span>BLK</span>
      </div>
      {players.map((player) => (
        <div className="player-stat-row" key={player.name}>
          <span>{player.name}</span>
          <span>{player.points}</span><span>{player.rebounds}</span><span>{player.assists}</span><span>{player.fieldGoals}</span>
          <span>{player.threePointers}</span><span>{player.steals}</span><span>{player.blocks}</span>
        </div>
      ))}
    </>
  );
}

function BaseballPlayerRows({players, hasStarted}: {players: LiveGameStats['away']; hasStarted: boolean}) {
  const pitchers = players.filter((player) => player.role === 'P');
  const batters = players.filter((player) => player.role !== 'P');

  return (
    <div className="mlb-player-sections">
      {pitchers.length > 0 && (
        <div className="mlb-player-section">
          <div className="mlb-pitcher-stat-header">
            <span>Pitcher</span><span>IP</span><span>PC-ST</span><span>ER</span><span>K</span><span>BB</span><span>ERA</span>
          </div>
          {pitchers.map((player) => (
            <div className="mlb-pitcher-stat-row" key={player.playerId ?? player.name}>
              <span>{player.name}</span>
              <span>{player.inningsPitched || '-'}</span>
              <span>{player.pitchCountStrikes || '-'}</span>
              <span>{player.earnedRuns ?? 0}</span>
              <span>{player.pitchingStrikeOuts ?? 0}</span>
              <span>{player.pitchingWalks ?? 0}</span>
              <span>{player.era || '-.--'}</span>
            </div>
          ))}
        </div>
      )}

      {batters.length > 0 && (
        <div className="mlb-player-section batters-start">
          <div className={`mlb-batter-stat-header ${hasStarted ? 'live' : ''}`}>
            <span>Batter</span><span>{hasStarted ? 'H-AB' : 'AVG'}</span><span>R</span><span>HR</span><span>RBI</span><span>BB</span><span>K</span>{!hasStarted && <span>OPS</span>}
          </div>
          {batters.map((player) => (
            <div className={`mlb-batter-stat-row ${hasStarted ? 'live' : ''}`} key={player.playerId ?? player.name}>
              <span>{player.name}</span>
              <span>{hasStarted ? player.battingLine || '-' : player.avg || '-'}</span>
              <span>{player.runs ?? 0}</span>
              <span>{player.homeRuns ?? 0}</span>
              <span>{player.rbi ?? 0}</span>
              <span>{player.walks ?? 0}</span>
              <span>{player.strikeOuts ?? 0}</span>
              {!hasStarted && <span>{player.ops || '-'}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function LivePlayerStats({game, league, date}: LivePlayerStatsProps) {
  const [result, setResult] = useState<{gameId: string; stats?: LiveGameStats; message?: string} | null>(null);
  const gameId = game?.gameId;

  useEffect(() => {
    if (!gameId) return;

    let cancelled = false;
    const shouldRefresh = game.isLive || game.status === 2;
    const loadStats = () => {
      const params = new URLSearchParams({
        league,
        date,
        awayTeam: game.awayFullName || game.away,
        homeTeam: game.homeFullName || game.home,
        awayTeamId: String(game.awayTeamId),
        homeTeamId: String(game.homeTeamId),
      });
      fetch(`${API_BASE}/players/${gameId}?${params.toString()}`)
        .then((response) => response.json())
        .then((data) => {
          if (cancelled) return;
          setResult(data.status === 'success'
            ? {gameId, stats: data}
            : {gameId, message: 'Live player stats are unavailable.'});
        })
        .catch(() => !cancelled && setResult({gameId, message: 'Live player stats are unavailable.'}));
    };

    loadStats();
    if (!shouldRefresh) {
      return () => {
        cancelled = true;
      };
    }
    const interval = window.setInterval(loadStats, REFRESH_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [gameId, league, date, game?.awayFullName, game?.away, game?.awayTeamId, game?.homeFullName, game?.home, game?.homeTeamId, game?.isLive, game?.status]);

  if (!game) return null;
  if (result?.gameId !== game.gameId) return <div className="team-leaders-message">Loading live player stats...</div>;
  if (result.message) return <div className="team-leaders-message">{result.message}</div>;
  if (!result.stats) return null;
  const stats = result.stats;
  if (!stats.hasStarted && stats.away.length === 0 && stats.home.length === 0) {
    return <div className="team-leaders-message">Player averages are unavailable.</div>;
  }

  const teams = [
    {name: game.away, logo: game.awayLogo, players: stats.away},
    {name: game.home, logo: game.homeLogo, players: stats.home},
  ];
  const isMlb = league === 'mlb' || stats.sport === 'mlb';
  const label = stats.hasStarted ? 'Players' : isMlb ? 'Season Leaders' : 'Season Averages';

  return (
    <section className="team-leaders" aria-label={stats.hasStarted ? 'Live player statistics' : 'Player season averages'}>
      {teams.map((team) => (
        <div className="team-leader-list" key={team.name}>
          <div className="team-leader-heading">
            {team.logo && <img src={team.logo} alt="" onError={hideBrokenLogo} />}
            <strong>{team.name} {label}</strong>
          </div>
          {isMlb
            ? <BaseballPlayerRows players={team.players} hasStarted={stats.hasStarted} />
            : <BasketballPlayerRows players={team.players} />}
        </div>
      ))}
    </section>
  );
}
