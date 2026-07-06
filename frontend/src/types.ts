export type League = 'nba' | 'wnba';

export type MoneylineSide = {
  team: string;
  fullName: string;
  price: number;
};

export type OddsData = {
  away: MoneylineSide;
  home: MoneylineSide;
  bookmakerTitle?: string;
  lastUpdate?: string;
};

export type LivePlayerStat = {
  name: string;
  points: number;
  rebounds: number;
  assists: number;
  fieldGoals: string;
  threePointers: string;
  steals: number;
  blocks: number;
};

export type LiveGameStats = {
  hasStarted: boolean;
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
  clock?: string;
  period?: number;
  isLive?: boolean;
  isFinal?: boolean;
};
