"""Score saved predictions without inference, or explicitly run the official harness.

Run from the repository root. Neither organizer script is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Command failed: " + " ".join(command))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred", type=Path, help="Score existing predictions; never decode video")
    source.add_argument("--videos", type=Path, help="Explicitly run the official harness first")
    parser.add_argument("--gt", type=Path, default=Path("annotations/dev_ground_truth.json"))
    parser.add_argument("--team", default="zeroth-law")
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--note", default="")
    parser.add_argument("--allow-run-errors", action="store_true",
                        help="Record failed/partial harness outputs for diagnosis, flagged in the experiment")
    args = parser.parse_args(argv)
    try:
        # Fail before expensive inference if annotations or configs are missing.
        for path in [args.gt, *args.config, *([args.pred] if args.pred else [])]:
            if not path.is_file():
                raise ValueError(f"Required file missing: {path}")
        if args.videos and not args.videos.exists():
            raise ValueError(f"Video input missing: {args.videos}")
        gt = json.loads(args.gt.read_text(encoding="utf-8"))
        if not isinstance(gt, dict) or not gt:
            raise ValueError("Ground truth must be a nonempty official video mapping")
        for video, info in gt.items():
            if not isinstance(info, dict) or not isinstance(info.get("events"), list):
                raise ValueError(f"Missing ground-truth events for {video}")
            duration = info.get("duration")
            if not isinstance(duration, (float, int)) or isinstance(duration, bool) or not 0 < duration < float("inf"):
                raise ValueError(f"Invalid ground-truth duration for {video}")

        import importlib.util
        spec = importlib.util.spec_from_file_location("official_evaluator", ROOT / "evaluate.py")
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        # The official validator normally validates predictions, not GT itself.
        gt_as_pred = {"team": "gt-format-check", "videos": {
            name: {"events": entry["events"], "risk": []} for name, entry in gt.items()
        }}
        errors, _ = official.validate(gt_as_pred, gt)
        if errors:
            raise ValueError("Invalid ground truth: " + "; ".join(errors))

        args.experiments.mkdir(parents=True, exist_ok=True)
        # Temporary outputs are retained in each successful experiment record.
        with tempfile.TemporaryDirectory(prefix="zlt-eval-") as temporary:
            work = Path(temporary)
            pred_path = args.pred
            if args.videos:
                pred_path = work / "predictions.json"
                run([sys.executable, "-B", str(ROOT / "run_submission.py"),
                     "--videos", str(args.videos.resolve()), "--out", str(pred_path), "--team", args.team])
            pred = json.loads(pred_path.read_text(encoding="utf-8"))
            run_errors = {name: entry["errors"] for name, entry in pred.get("log", {}).items()
                          if entry.get("errors")}
            if run_errors and not args.allow_run_errors:
                raise ValueError("Harness recorded failures; use --allow-run-errors only for diagnosis: "
                                 + json.dumps(run_errors))
            report_path = work / "report.json"
            run([sys.executable, "-B", str(ROOT / "evaluate.py"), "--pred", str(pred_path.resolve()),
                 "--gt", str(args.gt.resolve()), "--per-video", "--json", str(report_path)])
            record = {
                "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain"),
                "mode": "official-runner" if args.videos else "saved-predictions",
                "ground_truth": {"file": args.gt.name, "sha256": digest(args.gt)},
                "predictions_sha256": digest(pred_path),
                "evaluator_sha256": digest(ROOT / "evaluate.py"),
                "runner_sha256": digest(ROOT / "run_submission.py"),
                "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(
                    [ROOT / "solution.py", ROOT / "requirements.txt",
                     *(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")])},
                "configs": [{"file": p.name, "sha256": digest(p)} for p in args.config],
                "run_errors": run_errors, "note": args.note,
                "report": json.loads(report_path.read_text()),
                "predictions": pred,
            }
            # Exclusive creation avoids overwriting an earlier experiment.
            for number in range(1, 1000000):
                out = args.experiments / f"EXP-{number:03d}.json"
                record["id"] = out.stem
                serialized = json.dumps(record, indent=2, allow_nan=False) + "\n"
                try:
                    with out.open("x", encoding="utf-8") as file:
                        file.write(serialized)
                    print(f"Recorded {out}")
                    break
                except FileExistsError:
                    continue
            else:
                raise ValueError("Experiment number range exhausted")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"dev_eval: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
