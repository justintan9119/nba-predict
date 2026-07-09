export type League = 'nba' | 'wnba' | 'mlb';

export type PredictionMetric = {
  label: string;
  home: number;
  away: number;
  edge: number;
  leader: 'home' | 'away';
};

export type MoneylineSide = {
  team: string;
  fullName: string;
  price: number;
  ticker?: string;
  kalshiPrice?: number;
};

export type KalshiUserBet = {
  type: 'position' | 'order';
  orderId?: string;
  ticker: string;
  side?: 'away' | 'home';
  team?: string;
  contractSide?: string;
  action?: string;
  contracts?: number;
  remainingContracts?: number;
  netPosition?: number;
  totalTraded?: number;
  marketExposureCents?: number;
  priceCents?: number;
  costCents?: number | null;
  maxCostCents?: number | null;
  potentialProfitCents?: number | null;
  potentialPayoutCents?: number | null;
  status?: string;
  createdTime?: string;
};

export type OddsData = {
  away: MoneylineSide;
  home: MoneylineSide;
  bookmakerTitle?: string;
  lastUpdate?: string;
  userBets?: KalshiUserBet[];
  userBetsError?: string;
};

export type KalshiRecord = {
  configured: boolean;
  startTime: string;
  wins: number;
  losses: number;
  trackedContracts: number;
  realizedPnlCents: number;
  markets: number;
};

export type PredictionProbabilities = {
  home: number;
  away: number;
};

export type LivePlayerStat = {
  playerId?: number | string;
  name: string;
  points: number;
  rebounds: number;
  assists: number;
  fieldGoals: string;
  threePointers: string;
  steals: number;
  blocks: number;
  position?: string;
  role?: 'B' | 'P';
  summary?: string;
  battingLine?: string;
  runs?: number;
  rbi?: number;
  homeRuns?: number;
  walks?: number;
  strikeOuts?: number;
  avg?: string;
  ops?: string;
  inningsPitched?: string;
  earnedRuns?: number;
  pitchingStrikeOuts?: number;
  pitchingWalks?: number;
  era?: string;
  pitchCount?: number;
  pitchStrikes?: number;
  pitchCountStrikes?: string;
};

export type LiveGameStats = {
  hasStarted: boolean;
  sport?: League;
  away: LivePlayerStat[];
  home: LivePlayerStat[];
};

export type LiveData = {
  winner: string;
  confidence?: number;
  home: string;
  away: string;
  homeFullName?: string;
  awayFullName?: string;
  score: string;
  clock: string;
  period: number;
  statusText?: string;
  isFinal?: boolean;
  metrics?: PredictionMetric[];
  analysis?: string;
  probabilities?: PredictionProbabilities;
  modelProbabilities?: PredictionProbabilities;
};

export type Game = {
  home: string;
  away: string;
  homeFullName?: string;
  awayFullName?: string;
  gameId: string;
  homeTeamId: number;
  awayTeamId: number;
  homeLogo?: string;
  awayLogo?: string;
  awayScore?: number;
  homeScore?: number;
  status?: number;
  statusText?: string;
  startTime?: string;
  clock?: string;
  period?: number;
  isLive?: boolean;
  isFinal?: boolean;
};
