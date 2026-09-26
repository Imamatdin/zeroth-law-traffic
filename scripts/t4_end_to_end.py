"""Kaggle: full unchanged harness on one T4, CPU video decode, default 3x limit.

python scripts/t4_end_to_end.py --videos /kaggle/input/traffic --out /kaggle/working/zlt-profile
Place the local checkpoint at weights/yolo11m.pt before starting. Never downloads.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def summarize(prediction):
    rows = []
    for name, timing in prediction["log"].items():
        duration = timing["duration"]
        rows.append(dict(video=name, duration_s=duration, part_a_s=timing.get("part_a_sec"),
                         part_b_s=timing.get("part_b_sec"), wall_s=timing["total_sec"],
                         ratio=timing["total_sec"] / duration, errors=timing["errors"],
                         risk_samples=len(prediction["videos"][name]["risk"]),
                         within_3x=timing["total_sec"] <= 3 * duration))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new directory to preserve measurements")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.videos = str(Path(args.videos).resolve())
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Enable a Kaggle T4; exactly one CUDA device must be visible")
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        raise RuntimeError(f"Expected T4, found {gpu}")
    if not (ROOT / "weights/yolo11m.pt").is_file():
        raise FileNotFoundError("Place the local checkpoint at weights/yolo11m.pt; no downloads")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True)
    report = dict(gpu=gpu, visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                  cpu=platform.processor(), logical_cores=os.cpu_count(), python=sys.version,
                  torch=torch.__version__, cuda=torch.version.cuda, time_factor=3, runs=[])
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        report["nvidia_smi"] = smi.stdout + smi.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = str(exc)
    payloads, raw_files = [], []
    for number in range(1, args.repeats + 1):
        target = args.out / f"predictions_{number}.json"
        cmd = [sys.executable, str(ROOT / "run_submission.py"), "--videos", args.videos,
               "--out", str(target), "--team", "zeroth-law", "--solution", str(ROOT / "solution.py")]
        print("Running:", cmd, flush=True)
        started = time.perf_counter()
        with (args.out / f"harness_{number}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        run = dict(command=cmd, exit_code=result.returncode, process_wall_s=time.perf_counter() - started)
        if target.exists():
            raw_files.append(target.read_bytes())
            pred = json.loads(raw_files[-1])
            payloads.append({key: pred[key] for key in ("team", "videos")})
            run["videos"] = summarize(pred)
            check = subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--pred", str(target),
                                    "--validate-only"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            run["validation_exit_code"] = check.returncode
            run["validation_output"] = check.stdout + check.stderr
        report["runs"].append(run)
        print(json.dumps(run, indent=2), flush=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["prediction_payloads_equal"] = (len(payloads) == args.repeats and all(p == payloads[0] for p in payloads))
    report["full_files_equal"] = len(raw_files) == args.repeats and all(p == raw_files[0] for p in raw_files)
    report["note"] = "Harness timing logs vary; identical empty outputs after timeout are not a pass. Ratios use rounded official durations/times. Process wall includes model import/warm-up."
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    success = all(r["exit_code"] == 0 and r.get("validation_exit_code") == 0 and
                  all(v["within_3x"] and not v["errors"] for v in r.get("videos", [])) for r in report["runs"])
    return 0 if success and report["prediction_payloads_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
