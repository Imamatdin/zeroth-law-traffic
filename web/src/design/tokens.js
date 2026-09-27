// Canvas code reads colours from the CSS custom properties in tokens.css so there is one source of truth.
let cache = null;

export function palette() {
  if (cache) return cache;
  const cs = getComputedStyle(document.documentElement);
  const get = (name) => cs.getPropertyValue(`--${name}`).trim();
  const names = [
    'paper', 'paper-deep', 'paper-edge', 'ink', 'ink-soft', 'ink-faint', 'rule',
    'wash-asphalt', 'wash-asphalt-far', 'wash-pavement', 'wash-plaza', 'wash-refuge', 'wash-concrete', 'wash-kerb',
    'wash-leaf', 'wash-leaf-deep', 'wash-grass', 'wash-building', 'wash-building-deep', 'wash-glass',
    'wash-window-lit', 'wash-bus', 'wash-sign', 'line-paint', 'line-yellow',
    'sig-red', 'sig-yellow', 'sig-green', 'sig-unread', 'risk', 'field',
    'ev-jaywalking', 'ev-stop_line', 'ev-red_light', 'ev-stopped_vehicle', 'ev-wrong_way', 'ev-default',
  ];
  cache = Object.fromEntries(names.map((n) => [n, get(n)]));
  return cache;
}

export const eventColor = (label) => palette()[`ev-${label}`] || palette()['ev-default'];
export const eventVar = (label) => `var(--ev-${label}, var(--ev-default))`;

export const FONTS = {
  hand: "'Caveat', 'Segoe Print', cursive",
  ui: "'Recursive', system-ui, sans-serif",
};

// Vehicle body colours, weighted towards what the camera actually shows (mostly white and dark cars).
export const BODY_COLOURS = ['#f1f0ea', '#f1f0ea', '#e8e6df', '#3a3e4c', '#3a3e4c', '#a9adb3', '#7f9fc2',
  '#c9544f', '#d6bf73', '#8fae96', '#f1f0ea', '#5d6273'];
export const COAT_COLOURS = ['#c9544f', '#3e5f9c', '#d6a93a', '#2c877b', '#f1f0ea', '#3a3e4c', '#9a78b4', '#e38b6d',
  '#6f8f5e', '#f1f0ea'];
