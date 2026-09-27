import { useEffect, useRef } from 'react';
import { Diorama } from '../scene/renderer.js';

// The canvas plus its animation loop and pointer handling. UI state arrives through `uiRef` so the loop
// never restarts when toggles change.
export default function DioramaCanvas({ data, clock, uiRef, onSelect, rendererRef, reducedMotion, label }) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const r = new Diorama(canvas, data, { reducedMotion });
    rendererRef.current = r;
    const fit = () => {
      const rect = wrapRef.current.getBoundingClientRect();
      r.resize(rect.width, rect.height, Math.min(2, window.devicePixelRatio || 1));
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(wrapRef.current);
    let raf = 0;
    let last = performance.now();
    const loop = (now) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      clock.tick(dt);
      if (uiRef.current.layers.photo) r.loadPhoto();
      r.frame(clock.t, dt, uiRef.current);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [data, clock, uiRef, rendererRef, reducedMotion]);

  useEffect(() => {
    const el = canvasRef.current;
    let drag = null;
    const pos = (e) => {
      const b = el.getBoundingClientRect();
      return [e.clientX - b.left, e.clientY - b.top];
    };
    const down = (e) => {
      drag = { x: e.clientX, y: e.clientY, moved: 0, id: e.pointerId, touch: e.pointerType === 'touch' };
    };
    const move = (e) => {
      const r = rendererRef.current;
      if (!r) return;
      if (drag && drag.id === e.pointerId) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag.moved += Math.abs(dx) + Math.abs(dy);
        if (drag.moved > 6) {
          if (!drag.captured) {
            el.setPointerCapture(e.pointerId);
            drag.captured = true;
          }
          r.panBy(dx, drag.touch ? 0 : dy);
        }
        drag.x = e.clientX;
        drag.y = e.clientY;
      } else if (e.pointerType === 'mouse') {
        const [mx, my] = pos(e);
        el.style.cursor = r.hit(mx, my) != null ? 'pointer' : 'grab';
      }
    };
    const up = (e) => {
      const r = rendererRef.current;
      if (drag && drag.moved <= 6 && r) {
        const [mx, my] = pos(e);
        onSelect(r.hit(mx, my));
      }
      drag = null;
    };
    const wheel = (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const [mx, my] = pos(e);
      rendererRef.current?.zoomBy(Math.exp(-e.deltaY * 0.004), mx, my);
    };
    el.addEventListener('pointerdown', down);
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', () => (drag = null));
    el.addEventListener('wheel', wheel, { passive: false });
    return () => {
      el.removeEventListener('pointerdown', down);
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('wheel', wheel);
    };
  }, [onSelect, rendererRef]);

  return (
    <div className="diorama" ref={wrapRef}>
      <canvas ref={canvasRef} role="img" aria-label={label} />
    </div>
  );
}
