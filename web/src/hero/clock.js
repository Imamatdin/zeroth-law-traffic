// Replay clock shared by the canvas loop, the timeline and the notebook. Kept outside React state so the
// 60 fps loop does not re-render components; React views subscribe at a lower rate.
import { useEffect, useState } from 'react';

export function createClock(duration, { t = 0, playing = true } = {}) {
  const subs = new Set();
  const c = {
    t,
    playing,
    speed: 1,
    duration,
    set(patch) {
      Object.assign(c, patch);
      c.t = Math.max(0, Math.min(c.duration, c.t));
      subs.forEach((f) => f(c));
    },
    seek(t) {
      c.set({ t });
    },
    tick(dt) {
      if (!c.playing) return;
      let t = c.t + dt * c.speed;
      if (t >= c.duration) t = 0;
      c.t = t;
      subs.forEach((f) => f(c));
    },
    subscribe(f) {
      subs.add(f);
      return () => subs.delete(f);
    },
  };
  return c;
}

// Re-render at most `hz` times a second with a snapshot of the clock.
export function useClock(clock, hz = 8) {
  const [snap, setSnap] = useState(() => ({ t: clock.t, playing: clock.playing, speed: clock.speed }));
  useEffect(() => {
    let last = 0;
    let timer = null;
    const push = () => setSnap({ t: clock.t, playing: clock.playing, speed: clock.speed });
    const unsub = clock.subscribe((c) => {
      const now = performance.now();
      if (now - last > 1000 / hz || !c.playing) {
        last = now;
        push();
      } else if (!timer) {
        timer = setTimeout(() => {
          timer = null;
          last = performance.now();
          push();
        }, 1000 / hz);
      }
    });
    return () => {
      unsub();
      clearTimeout(timer);
    };
  }, [clock, hz]);
  return snap;
}

export const fmtT = (t) => `${t.toFixed(1)} s`;
