"""Adapters matching zlt-web's C3905 export documents (web commit 7f01d46).

Result is an envelope: replay/events/risk/signal retain the export shapes;
metadata and detections are separate, explicitly documented extensions.
"""
import ast
from dataclasses import asdict
from pathlib import Path

import numpy as np
from src.contracts import OBJECT_CLASSES
from src.scene.signal import smooth_states

Q = 10000
ROOT = Path(__file__).resolve().parents[1]


def classes():
    for n in ast.parse((ROOT / 'solution.py').read_text()).body:
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
        if any(getattr(t, 'id', None) == 'CLASSES' for t in targets):
            return ast.literal_eval(n.value)
    raise RuntimeError('Official classes missing')


def replay_document(tracks, world, times, meta):
    out = []
    index = {frame: k for k, (frame, _) in enumerate(times)}
    stats = {(int(r.frame), int(r.track_id)): r for r in world.itertuples()} if len(world) else {}
    mapping = {}
    for tid, g in tracks.groupby('track_id', sort=True):
        g = g.sort_values('frame')
        raws = sorted(int(x) for x in g.raw_track_id.unique())
        for raw in raws:
            mapping[raw] = int(tid)
        rec = dict(id=int(tid), cls=int(g.cls.mode().iloc[0]), raw=raws,
                   k=[], x=[], y=[], w=[], h=[], v=[], hd=[], st=[])
        for r in g.itertuples():
            s = stats.get((int(r.frame), int(tid)))
            rec['k'].append(index[int(r.frame)])
            for key, val, denom in [('x', r.fx, meta['width']), ('y', r.fy, meta['height']),
                                    ('w', r.x2-r.x1, meta['width']), ('h', r.y2-r.y1, meta['height'])]:
                rec[key].append(round(val / denom * Q))
            rec['v'].append(round(s.speed_rel * 100) if s else 0)
            rec['hd'].append(round(s.heading) if s else 0)
            rec['st'].append(round(s.stationary_s * 10) if s else 0)
        out.append(rec)
    return dict(video=meta['video_id'], fps=meta['fps'], duration=meta['duration'],
        width=meta['width'], height=meta['height'], q=Q, t=[round(t, 3) for _, t in times],
        classes=list(OBJECT_CLASSES), tracks=out,
        units={'x,y': 'foot point (bottom-centre of the box), normalized x Q',
               'w,h': 'box size, normalized x Q', 'v': 'speed_rel x 100 (box heights per second)',
               'hd': 'image-space heading, degrees, y down', 'st': 'stationary seconds x 10',
               'k': 'index into t'}), mapping


def event_document(raw, segments, config, matched):
    def event(e):
        d = asdict(e)
        return dict(label=d['label'], start=round(d['start'], 3), end=round(d['end'], 3),
                    confidence=d['confidence'], track_ids=list(d['track_ids']), evidence=d['evidence'])
    enabled = {k: bool(v.get('enabled')) for k, v in config.items()
               if isinstance(v, dict) and 'enabled' in v}
    from src.postprocess.segments import to_official
    return dict(status='scene-dependent events skipped: camera mismatch' if not matched else 'demo profile',
                mode='enabled submission engines only', classes=classes(), enabled_in_submission=enabled,
                submitted=to_official(segments), raw_events=[event(e) for e in raw],
                segments=[event(e) for e in segments])


def runs(states, times, end):
    out = []
    for state, t in zip(states, times):
        if out and out[-1][2] == state:
            continue
        if out:
            out[-1][1] = round(t, 3)
        out.append([round(t, 3), round(end, 3), state])
    return out


def signal_document(rows, duration):
    signals = {}
    for sid in sorted({r['signal'] for r in rows}):
        rr = [r for r in rows if r['signal'] == sid]
        times, raw = [r['t'] for r in rr], [r['state'] for r in rr]
        smooth = smooth_states(raw)
        bridge = runs(['bridged' if a != b else a for a, b in zip(raw, smooth)], times, duration)
        signals[sid] = dict(raw=runs(raw, times, duration), smoothed=runs(smooth, times, duration),
                            bridged=[[a, b] for a, b, s in bridge if s == 'bridged'])
    return dict(signals=signals, near_stop_crossings=[],
                note='Raw sampled lamp readings and Part A smoothing. Stop-line crossing diagnostics are not computed by the demo.')


def risk_document(curve, evidence, poses, mapping, meta, matched):
    from evaluate import alarm_starts
    pairs = []
    for i, ev in enumerate(evidence):
        if 'pair' not in ev:
            continue
        a, b = ev['pair']
        point = poses[i]
        if point is None:
            continue
        pairs.append(dict(i=i, a=mapping.get(a), b=mapping.get(b), raw_ids=[a, b], cls=ev['cls'],
            tca=ev['tca_s'], dmin_h=ev['dmin_h'], closing_h=ev['closing_h_per_s'],
            dist_h=ev['dist_h'], held=ev['course_held_s'], flags=ev['flags'],
            px=round(float(np.clip(point[0] / meta['width'], -.2, 1.2)) * Q),
            py=round(float(np.clip(point[1] / meta['height'], -.2, 1.2)) * Q)))
    def percentiles(values):
        return {str(q): round(float(np.percentile(values, q)), 4) if values else 0.0
                for q in (50, 90, 99, 99.9, 100)}
    raw, risk = [e['raw'] for e in evidence], [s for _, s in curve]
    return dict(status='demo profile; camera priors active' if matched else 'generic geometry only; uncalibrated for this scene',
        threshold=.5, horizon_s=5., alarm_starts=alarm_starts(curve),
        percentiles=dict(raw=percentiles(raw), risk=percentiles(risk)),
        t=[round(t, 3) for t, _ in curve], risk=risk, raw=raw,
        smoothed=[e['smoothed'] for e in evidence], pairs=pairs,
        note='Causal demo risk from shared current-frame observations, before offline stitching. Scores are demo estimates, not validated accident probabilities. '
             'Pair ids map to Part A display tracks where possible; px/py is a display extrapolation.')
