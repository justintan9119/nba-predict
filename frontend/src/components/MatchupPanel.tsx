import {LivePlayerStats} from './LivePlayerStats';
import {teamColor} from '../teamColors';
import type {Game, KalshiUserBet, League, LiveData, MoneylineSide, OddsData, PredictionProbabilities} from '../types';
import {formatAmericanOdds, formatClock, formatPercent, impliedProbabilityFromAmericanOdds} from '../utils';

type MatchupPanelProps = {
  loading: boolean;
  league: League;
  date: string;
  liveData: LiveData | null;
  winner?: string;
  confidence: number | string;
  predictionMessage: string;
  selectedGame?: Game;
  odds: OddsData | null;
  oddsLoading: boolean;
  oddsEnabled: boolean;
  oddsMessage: string;
  kalshiBetMessage: string;
  predictionProbabilities: PredictionProbabilities | null;
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

function formatCents(value?: number | null) {
  if (value === null || value === undefined) {
    return '--';
  }
  return `$${(value / 100).toFixed(2)}`;
}

function BetDescription({bet}: {bet: KalshiUserBet}) {
  const contractSide = bet.contractSide || 'YES';
  if (bet.type === 'position') {
    const contractText = `${bet.contracts ?? 0} ${contractSide} contract${bet.contracts === 1 ? '' : 's'}`;
    const exposure = bet.marketExposureCents ? `Exposure ${formatCents(bet.marketExposureCents)}` : `Total traded ${bet.totalTraded ?? 0}`;
    return <>{contractText}<em>{exposure}</em></>;
  }

  const remaining = bet.remainingContracts ?? bet.contracts ?? 0;
  const price = bet.priceCents ? ` @ ${bet.priceCents}¢` : '';
  return <>{bet.action || 'BUY'} {remaining} {contractSide}{price}<em>Max cost {formatCents(bet.maxCostCents)}</em></>;
}

function UserBets({bets}: {bets?: KalshiUserBet[]}) {
  if (!bets?.length) {
    return null;
  }

  return (
    <div className="kalshi-user-bets">
      <div className="kalshi-user-bets-heading">Your Kalshi Bets</div>
      {bets.map((bet, index) => (
        <div className="kalshi-user-bet" key={`${bet.ticker}-${bet.type}-${index}`}>
          <span style={{color: teamColor(bet.team || '')}}>{bet.team || bet.ticker}</span>
          <strong>{bet.type === 'position' ? 'Position' : 'Resting order'}</strong>
          <small><BetDescription bet={bet} /></small>
        </div>
      ))}
    </div>
  );
}

function OddsDisplay({odds}: {odds: OddsData}) {
  return (
    <div className="odds-card">
      <div className="odds-heading">
        <span>Kalshi Market Prices</span>
        {odds.bookmakerTitle && <em>{odds.bookmakerTitle}</em>}
      </div>
      <OddsSide side={odds.away} />
      <OddsSide side={odds.home} />
      <UserBets bets={odds.userBets} />
      {odds.userBetsError && <div className="odds-message">Kalshi bet lookup failed: {odds.userBetsError}</div>}
      {odds.lastUpdate && <div className="odds-updated">Updated {new Date(odds.lastUpdate).toLocaleString()}</div>}
    </div>
  );
}

function BettingEdge({odds, probabilities, isLive = false}: {odds: OddsData; probabilities: PredictionProbabilities; isLive?: boolean}) {
  const rawAwayProbability = impliedProbabilityFromAmericanOdds(odds.away.price);
  const rawHomeProbability = impliedProbabilityFromAmericanOdds(odds.home.price);
  const vigTotal = rawAwayProbability !== null && rawHomeProbability !== null
    ? rawAwayProbability + rawHomeProbability
    : null;
  const fairProbabilities = vigTotal && vigTotal > 0
    ? {
        away: rawAwayProbability === null ? null : rawAwayProbability / vigTotal,
        home: rawHomeProbability === null ? null : rawHomeProbability / vigTotal,
      }
    : {away: null, home: null};
  const sides = [
    {
      key: 'away',
      label: odds.away.fullName,
      team: odds.away.team,
      price: odds.away.price,
      modelProbability: probabilities.away,
      fairProbability: fairProbabilities.away,
    },
    {
      key: 'home',
      label: odds.home.fullName,
      team: odds.home.team,
      price: odds.home.price,
      modelProbability: probabilities.home,
      fairProbability: fairProbabilities.home,
    },
  ] as const;

  const scoredSides = sides.map((side) => {
    const decimalOdds = side.price > 0 ? 1 + side.price / 100 : 1 + 100 / Math.abs(side.price);
    const edge = side.fairProbability === null ? null : side.modelProbability - side.fairProbability;
    const expectedValue = (side.modelProbability * decimalOdds) - 1;
    const b = decimalOdds - 1;
    const q = 1 - side.modelProbability;
    const kelly = b > 0 ? Math.max(0, ((b * side.modelProbability) - q) / b) : null;
    return {...side, decimalOdds, edge, expectedValue, kelly, isModelPick: side.modelProbability >= 0.5};
  });

  const modelPick = scoredSides.find((side) => side.isModelPick);

  const worthBetting = Boolean(modelPick && (modelPick.edge ?? 0) >= 0.03 && modelPick.expectedValue > 0);

  return (
    <div className={`bet-edge-card ${worthBetting ? 'positive' : 'neutral'}`}>
      <div className="bet-edge-heading">
        <span>{isLive ? 'Live Betting Edge' : 'Betting Edge'}</span>
        <strong>{worthBetting ? 'Worth betting' : 'No clear bet'}</strong>
      </div>
      {modelPick && (
        <div className="bet-edge-summary">
          {worthBetting
            ? <>Model side: <strong style={{color: teamColor(modelPick.team)}}>{modelPick.label}</strong> {formatAmericanOdds(modelPick.price)}</>
            : <>Model side has no positive bet edge: <strong style={{color: teamColor(modelPick.team)}}>{modelPick.label}</strong> {formatAmericanOdds(modelPick.price)}</>}
        </div>
      )}
      <div className="bet-edge-table">
        <div className="bet-edge-row bet-edge-header">
          <span>Side</span><span>Odds</span><span>Model</span><span>Fair</span><span>Edge</span><span>EV/$1</span><span>Kelly</span>
        </div>
        {scoredSides.map((side) => (
          <div className="bet-edge-row" key={side.key}>
            <span style={{color: teamColor(side.team)}}>{side.label}</span>
            <span>{formatAmericanOdds(side.price)}</span>
            <span>{formatPercent(side.modelProbability)}</span>
            <span>{formatPercent(side.fairProbability)}</span>
            <span className={side.isModelPick && (side.edge ?? 0) >= 0 ? 'positive-edge' : 'negative-edge'}>{side.isModelPick ? formatPercent(side.edge, 1) : '--'}</span>
            <span className={side.isModelPick && side.expectedValue >= 0 ? 'positive-edge' : 'negative-edge'}>{side.isModelPick ? `${side.expectedValue >= 0 ? '+' : ''}${side.expectedValue.toFixed(3)}` : '--'}</span>
            <span>{side.isModelPick ? formatPercent(side.kelly, 1) : '--'}</span>
          </div>
        ))}
      </div>
      <div className="odds-message">
        Edge, EV, and Kelly are shown only for the side the model projects to win. Edge uses de-vigged fair market probability; EV uses the actual Kalshi price for expected profit per $1 staked.
      </div>
    </div>
  );
}

function LiveMatchup({liveData, selectedGame, league}: {liveData: LiveData; selectedGame?: Game; league: League}) {
  const isFinal = Boolean(liveData.isFinal);
  const clockLabel = league === 'mlb' ? liveData.clock : `Quarter ${liveData.period} - ${formatClock(liveData.clock)}`;
  return (
    <div className="live-container" style={{borderColor: isFinal ? '#555' : '#ff4444'}}>
      <div className="live-badge" style={{backgroundColor: isFinal ? '#555' : '#ff4444'}}>{isFinal ? 'FINAL' : 'LIVE'}</div>
      <div className="live-score-container">
        <TeamName name={liveData.away} fullName={liveData.awayFullName} logoUrl={selectedGame?.awayLogo} />
        <span className="live-score">{liveData.score}</span>
        <TeamName name={liveData.home} fullName={liveData.homeFullName} logoUrl={selectedGame?.homeLogo} />
      </div>
      <div className="live-clock">{isFinal ? 'Game Ended' : clockLabel}</div>
      <div className="win" style={{marginTop: '1rem', color: isFinal ? '#fff' : '#646cff'}}>
        {isFinal ? null : liveData.confidence === undefined
          ? 'Live score is in progress'
          : <>Live Projection: <strong>{liveData.winner}</strong> is favored (<strong>{liveData.confidence}%</strong>)</>}
      </div>
    </div>
  );
}

function PregameMatchup({
  winner, confidence, selectedGame, league, predictionMessage,
}: Pick<MatchupPanelProps, 'winner' | 'confidence' | 'selectedGame' | 'league' | 'predictionMessage'>) {
  if (!winner) {
    return <div>{predictionMessage || 'Select a game to see prediction'}</div>;
  }

  return (
    <div className="live-container" style={{borderColor: '#646cff'}}>
      <div className="live-badge" style={{backgroundColor: '#646cff'}}>PRE-GAME</div>
      <div className="live-score-container">
        <TeamName name={selectedGame?.away || ''} fullName={selectedGame?.awayFullName} logoUrl={selectedGame?.awayLogo} />
        <span className="live-score">VS</span>
        <TeamName name={selectedGame?.home || ''} fullName={selectedGame?.homeFullName} logoUrl={selectedGame?.homeLogo} />
      </div>
      <div className="live-clock">{league === 'mlb' ? 'Advanced Metric Analysis' : 'Matchup Analysis'}</div>
      <div className="win" style={{marginTop: '1rem'}}>
        Projection: <strong>{winner}</strong> is favored to win (<strong>{confidence}%</strong>)
      </div>
    </div>
  );
}

function MatchupDetails({
  selectedGame, league, date, liveData, odds, oddsLoading, oddsEnabled, oddsMessage, predictionProbabilities, onToggleOddsApi,
  kalshiBetMessage,
}: MatchupPanelProps) {
  return (
    <div className="matchup-details">
      <div className="odds-api-control">
        <button className={oddsEnabled ? '' : 'disabled'} onClick={onToggleOddsApi}>
          {oddsEnabled ? 'Test: Disable Kalshi API' : 'Test: Enable Kalshi API'}
        </button>
        {!oddsEnabled && <span>Kalshi API disabled</span>}
      </div>

      {selectedGame && <LivePlayerStats game={selectedGame} league={league} date={date} />}
      {odds && <OddsDisplay odds={odds} />}
      {odds && predictionProbabilities && <BettingEdge odds={odds} probabilities={predictionProbabilities} isLive={Boolean(liveData)} />}
      {oddsLoading && <div className="odds-message">Loading Kalshi prices...</div>}
      {oddsMessage && <div className="odds-message">{oddsMessage}</div>}
      {kalshiBetMessage && <div className="odds-message">{kalshiBetMessage}</div>}
    </div>
  );
}

export function MatchupPanel(props: MatchupPanelProps) {
  return (
    <div className="right-side">
      <div className="matchup-sticky">
        {props.loading
          ? <div className="loader">{props.predictionMessage || 'Analyzing matchup data...'}</div>
          : props.liveData
            ? <LiveMatchup liveData={props.liveData} selectedGame={props.selectedGame} league={props.league} />
            : <PregameMatchup winner={props.winner} confidence={props.confidence} selectedGame={props.selectedGame} league={props.league} predictionMessage={props.predictionMessage} />}
      </div>
      <MatchupDetails {...props} />
    </div>
  );
}
