import {LivePlayerStats} from './LivePlayerStats';
import {teamColor} from '../teamColors';
import type {Game, League, LiveData, MoneylineSide, OddsData} from '../types';
import {formatAmericanOdds, formatClock} from '../utils';

type MatchupPanelProps = {
  loading: boolean;
  league: League;
  date: string;
  liveData: LiveData | null;
  winner?: string;
  confidence: number | string;
  selectedGame?: Game;
  odds: OddsData | null;
  oddsLoading: boolean;
  oddsEnabled: boolean;
  oddsMessage: string;
  onToggleOddsApi: () => void;
};

function TeamName({name, fullName, logoUrl}: {name: string; fullName?: string; logoUrl?: string}) {
  return <span className="team-name" style={{color: teamColor(name)}}>{logoUrl && <img src={logoUrl} alt="" />}{fullName || name}</span>;
}

function OddsSide({side}: {side: MoneylineSide}) {
  const color = teamColor(side.team);
  return (
    <div className="odds-row" style={{borderColor: color}}>
      <span style={{color}}>{side.fullName}</span>
      <strong>{formatAmericanOdds(side.price)}</strong>
    </div>
  );
}

function OddsDisplay({odds}: {odds: OddsData}) {
  return (
    <div className="odds-card">
      <div className="odds-heading">
        <span>Moneyline Odds</span>
        {odds.bookmakerTitle && <em>{odds.bookmakerTitle}</em>}
      </div>
      <OddsSide side={odds.away} />
      <OddsSide side={odds.home} />
      {odds.lastUpdate && <div className="odds-updated">Updated {new Date(odds.lastUpdate).toLocaleString()}</div>}
    </div>
  );
}

function LiveMatchup({liveData, selectedGame, league, date}: {liveData: LiveData; selectedGame?: Game; league: League; date: string}) {
  const isFinal = Boolean(liveData.isFinal);
  return (
    <div className="live-container" style={{borderColor: isFinal ? '#555' : '#ff4444'}}>
      <div className="live-badge" style={{backgroundColor: isFinal ? '#555' : '#ff4444'}}>{isFinal ? 'FINAL' : 'LIVE'}</div>
      <div className="live-score-container">
        <TeamName name={liveData.away} fullName={liveData.awayFullName} logoUrl={selectedGame?.awayLogo} />
        <span className="live-score">{liveData.score}</span>
        <TeamName name={liveData.home} fullName={liveData.homeFullName} logoUrl={selectedGame?.homeLogo} />
      </div>
      <div className="live-clock">{isFinal ? 'Game Ended' : `Quarter ${liveData.period} - ${formatClock(liveData.clock)}`}</div>
      <div className="win" style={{marginTop: '1rem', color: isFinal ? '#fff' : '#646cff'}}>
        {isFinal ? null : liveData.confidence === undefined
          ? 'Live score is in progress'
          : <>Live Projection: <strong>{liveData.winner}</strong> is favored (<strong>{liveData.confidence}%</strong>)</>}
      </div>
      <LivePlayerStats game={selectedGame} league={league} date={date} />
    </div>
  );
}

function PregameMatchup({
  league, date, winner, confidence, selectedGame, odds, oddsLoading, oddsMessage,
}: Omit<MatchupPanelProps, 'loading' | 'liveData'>) {
  if (!winner) {
    return <div>Select a game to see prediction</div>;
  }

  return (
    <div className="live-container" style={{borderColor: '#646cff'}}>
      <div className="live-badge" style={{backgroundColor: '#646cff'}}>PRE-GAME</div>
      <div className="live-score-container">
        <TeamName name={selectedGame?.away || ''} fullName={selectedGame?.awayFullName} logoUrl={selectedGame?.awayLogo} />
        <span className="live-score">VS</span>
        <TeamName name={selectedGame?.home || ''} fullName={selectedGame?.homeFullName} logoUrl={selectedGame?.homeLogo} />
      </div>
      <div className="live-clock">Matchup Analysis</div>
      <div className="win" style={{marginTop: '1rem'}}>
        Projection: <strong>{winner}</strong> is favored to win (<strong>{confidence}%</strong>)
      </div>
      <LivePlayerStats game={selectedGame} league={league} date={date} />
      {odds && <OddsDisplay odds={odds} />}
      {oddsLoading && <div className="odds-message">Loading moneyline odds...</div>}
      {oddsMessage && <div className="odds-message">{oddsMessage}</div>}
    </div>
  );
}

export function MatchupPanel(props: MatchupPanelProps) {
  return (
    <div className="right-side">
      <div className="odds-api-control">
        <button className={props.oddsEnabled ? '' : 'disabled'} onClick={props.onToggleOddsApi}>
          {props.oddsEnabled ? 'Test: Disable Odds API' : 'Test: Enable Odds API'}
        </button>
        {!props.oddsEnabled && <span>Odds API disabled</span>}
      </div>
      {props.loading
        ? <div className="loader">Analyzing matchup data...</div>
        : props.liveData
          ? <LiveMatchup liveData={props.liveData} selectedGame={props.selectedGame} league={props.league} date={props.date} />
          : <PregameMatchup {...props} />}
    </div>
  );
}
