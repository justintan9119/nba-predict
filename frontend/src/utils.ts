import type {Game, LiveData} from './types';

export const dateInputValue = (date = new Date()) => {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return localDate.toISOString().slice(0, 10);
};

export const formatAmericanOdds = (value?: number) => {
  if (typeof value !== 'number') {
    return '--';
  }
  return value > 0 ? `+${value}` : `${value}`;
};

export const formatClock = (clock?: string) => {
  const match = clock?.match(/PT(\d+)M(\d+(?:\.\d+)?)S?/);
  if (!match) {
    return clock || 'LIVE';
  }
  return `${Number(match[1])}:${Math.floor(Number(match[2])).toString().padStart(2, '0')}`;
};

export const liveDataFromGame = (game: Game): LiveData => {
  const homeScore = game.homeScore ?? 0;
  const awayScore = game.awayScore ?? 0;
  return {
    winner: homeScore === awayScore ? '' : homeScore > awayScore ? game.home : game.away,
    isFinal: game.isFinal,
    home: game.home,
    away: game.away,
    homeFullName: game.homeFullName,
    awayFullName: game.awayFullName,
    score: `${awayScore} - ${homeScore}`,
    clock: game.isFinal ? 'FINAL' : formatClock(game.clock || game.statusText),
    period: game.period ?? 0,
    statusText: game.statusText,
  };
};
