"""Causal best-effort wall allocation; mandatory CPU decode cannot be skipped."""
import json
import math
import sys
import time
from collections import deque


def log(action, **fields):
    print(json.dumps({"runtime": action, **fields}, sort_keys=True), file=sys.stderr, flush=True)


class BudgetGuard:
    SHARES = {"A": 1.25, "B": 1.55}  # 0.2x remains for harness/uncertainty.
    DECODE = {"A": 0.85, "B": 1.40}  # Local measurements as priors, NOT judge measurements.
    MAX_STRIDE = 15
    WINDOW = 5

    def __init__(self, part, meta, clock=time.perf_counter):
        self.part, self.clock = part, clock
        self.n, self.fps = int(meta["n_frames"]), float(meta["fps"])
        self.duration = self.n / self.fps
        self.limit = self.SHARES[part] * self.duration
        self.started = clock()
        self.stride, self.next_frame = 3, 0
        self.cost, self.work, self.calls = 0.0, 0.0, 0
        self.costs = deque(maxlen=self.WINDOW)
        self.cap_reported = False

    def due(self, idx):
        return idx >= self.next_frame

    def observed(self, idx, seconds):
        self.calls += 1
        self.work += seconds
        self.costs.append(seconds)
        self.next_frame = idx + self.stride
        if len(self.costs) < self.WINDOW:
            return
        self.cost = sum(self.costs) / len(self.costs)
        elapsed = self.clock() - self.started
        progress = min(1.0, (idx + 1) / self.n)
        decode = self.DECODE[self.part] * self.duration
        if progress >= 0.10:
            decode = max(decode, max(0, elapsed - self.work) / progress)
        reserve = min(5.0, self.duration * 0.05) if self.part == "A" else 0.5
        allowance = self.limit - elapsed - decode * (1 - progress) - reserve
        remaining = max(0, self.n - idx - 1)
        desired = (math.ceil(remaining * self.cost / allowance / 3) * 3
                   if allowance > 0 else math.ceil((remaining + 3) / 3) * 3)
        capped = min(self.MAX_STRIDE, desired)
        if capped > self.stride:
            log("guard_widen", part=self.part, frame=idx, elapsed_s=round(elapsed, 4),
                old_stride=self.stride, stride=capped, requested_stride=desired,
                reason="rolling inference cost exceeds remaining allocation", window=len(self.costs),
                remaining_inference_s=round(allowance, 4), mean_call_s=round(self.cost, 4))
            self.stride = capped
        if desired > self.MAX_STRIDE and not self.cap_reported:
            log("guard_cap", part=self.part, frame=idx, stride=self.stride, requested_stride=desired,
                reason="preserve temporal coverage; budget overrun is preferable to stride above 15")
            self.cap_reported = True
        self.next_frame = idx + self.stride

    def finish(self):
        elapsed = self.clock() - self.started
        log("part_finished", part=self.part, wall_s=round(elapsed, 4),
            ratio=round(elapsed / self.duration, 5), limit_s=self.limit,
            over_budget=elapsed > self.limit, inference_calls=self.calls, final_stride=self.stride)
