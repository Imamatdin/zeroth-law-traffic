"""Turn rich per-class events into segments the official harness will keep unchanged.

The harness drops (not merges) later same-class overlaps and rounds to 3 decimals
after its validity check, so everything it would do is done here first, deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.contracts import Event


@dataclass(frozen=True)
class SegmentRules:
    merge_gap_s: float = 0.0
    min_dur_s: float = 0.0


def _union(a: Event, b: Event) -> Event:
    return replace(a, start=min(a.start, b.start), end=max(a.end, b.end),
                   confidence=max(a.confidence, b.confidence),
                   track_ids=tuple(sorted(set(a.track_ids) | set(b.track_ids))),
                   evidence={"merged": [a.evidence, b.evidence]})


def postprocess(events: list[Event], duration: float, fps: float,
                rules: dict[str, SegmentRules] | None = None) -> list[Event]:
    """Clamp, merge same-class overlaps and near gaps, drop blips; deterministic order."""
    if duration <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    rules = rules or {}
    frame = 1.0 / fps
    by_label: dict[str, list[Event]] = {}
    for ev in events:
        start, end = max(0.0, ev.start), min(ev.end, duration)
        if end - start < frame / 2:
            continue
        by_label.setdefault(ev.label, []).append(replace(ev, start=start, end=end))
    out = []
    for label in sorted(by_label):
        rule = rules.get(label, SegmentRules())
        merged: list[Event] = []
        for ev in sorted(by_label[label], key=lambda e: (e.start, e.end)):
            if merged and ev.start <= merged[-1].end + rule.merge_gap_s:
                merged[-1] = _union(merged[-1], ev)
            else:
                merged.append(ev)
        for ev in merged:
            if ev.end - ev.start < max(rule.min_dur_s, frame):
                # A one-frame floor keeps 3-decimal rounding from collapsing start == end.
                if ev.end - ev.start < rule.min_dur_s:
                    continue
                ev = replace(ev, end=min(duration, ev.start + frame))
            out.append(ev)
    return sorted(out, key=lambda e: (e.start, e.end, e.label))


def to_official(events: list[Event]) -> list[list]:
    """Round exactly like the harness; assumes postprocess() already ran."""
    rows = []
    last_end: dict[str, float] = {}
    for ev in sorted(events, key=lambda e: (e.label, e.start)):
        s, e = round(ev.start, 3), round(ev.end, 3)
        s = max(s, last_end.get(ev.label, 0.0))
        if s >= e:
            continue
        rows.append([s, e, ev.label])
        last_end[ev.label] = e
    return sorted(rows)
