const TEAM_COLORS: Record<string, string> = {
  Knicks: '#F58426', Spurs: '#C4CED4', Lakers: '#FDB927', Celtics: '#007A33',
  Warriors: '#FFC72C', Bulls: '#CE1141', Suns: '#E56020', Heat: '#98002E',
  '76ers': '#006BB6', Nets: '#FFFFFF', Bucks: '#EEE1AF', Cavaliers: '#860038',
  Mavericks: '#00538C', Nuggets: '#FEC524', Pistons: '#ED174C', Rockets: '#CE1141',
  Pacers: '#FDBB30', Clippers: '#C8102E', Grizzlies: '#5D76A9', Timberwolves: '#236192',
  Pelicans: '#C8102E', Thunder: '#007AC1', Magic: '#0077C0', Kings: '#5A2D81',
  'Trail Blazers': '#E03A3E', Jazz: '#002B5C', Hawks: '#E03A3E', Hornets: '#00788C',
  Raptors: '#CE1141', Wizards: '#002B5C',

  Dream: '#C8102E',
  Sky: '#418FDE',
  Sun: '#E56020',
  Wings: '#007DC3',
  Valkyries: '#60269E',
  Fever: '#FDB927',
  Aces: '#A71930',
  Sparks: '#702F8A',
  Lynx: '#236192',
  Liberty: '#6ECEB2',
  Mercury: '#E56020',
  Storm: '#2C5234',
  Mystics: '#002B5C',
};

const FALLBACK_COLOR = '#646cff';

export const teamColor = (team: string) => TEAM_COLORS[team] || FALLBACK_COLOR;

export const teamTextColor = (team: string) => {
  const color = teamColor(team).slice(1);
  const [red, green, blue] = [0, 2, 4].map((index) => Number.parseInt(color.slice(index, index + 2), 16));
  const brightness = (red * 299 + green * 587 + blue * 114) / 1000;
  return brightness > 155 ? '#111' : '#fff';
};
