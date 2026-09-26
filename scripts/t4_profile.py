"""Kaggle single-T4 YOLO11m@960 FP16 profiling, using real sample frames.

Notebook cell (fresh subprocess; attach repo, weights and sample as Kaggle inputs):
!python /kaggle/working/zeroth-law-traffic/scripts/t4_profile.py \
    --video /kaggle/input/traffic/C3905.MP4 \
    --weights /kaggle/input/weights/yolo11m.pt \
    --out /kaggle/working/t4_profile.json

Needs CUDA-enabled torch, torchvision, ultralytics, OpenCV, numpy. Does not install
packages or download weights. Decode/preload is CPU-only and reported separately.
Times batch-1 detector wrapper including preprocessing, transfers, NMS and box
conversion; synchronize CUDA at both ends. No tracking/rules in this measurement.
"""
from __future__ import annotations

import os
# Must precede torch/ultralytics imports; even on a two-T4 notebook use one GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("t4_profile.json"))
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    a = p.parse_args()
    if a.frames < 2 or a.repeats < 1 or a.warmup < 1:
        p.error("Need at least 2 frames, 1 repeat, and 1 warmup")
    if not a.video.is_file() or not a.weights.is_file():
        p.error("Video and local YOLO11m weights must exist (no auto-download)")
    if a.out.exists():
        p.error("Output exists; use a new --out")

    import cv2
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: enable Kaggle GPU T4 and install CUDA torch")
    gpu = torch.cuda.get_device_name(0)
    if torch.cuda.device_count() != 1 or "T4" not in gpu:
        raise RuntimeError(f"Expected exactly one visible T4, found {torch.cuda.device_count()}: {gpu}")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.perception.detector import UltralyticsDetector

    start = time.perf_counter()
    cap = cv2.VideoCapture(str(a.video), cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if n < a.frames or fps <= 0:
        raise RuntimeError("Insufficient frames or invalid video metadata")
    indices = np.linspace(0, n - 1, a.frames, dtype=int).tolist()
    wanted, frames = set(indices), []
    try:
        for i in range(n):
            if not cap.grab():
                raise RuntimeError(f"Decoder ended at {i}/{n}")
            if i in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(f"Cannot retrieve frame {i}")
                frames.append(frame)
    finally:
        cap.release()
    decode_sec = time.perf_counter() - start
    detector = UltralyticsDetector(a.weights.resolve(), imgsz=960, conf=0.1, iou=0.6,
                                    device="0", prescale_width=1920, half=True)
    # A assumes FFmpeg emits 1920 BGR; B receives full 4K BGR from the harness.
    # A's once-only pre-resize below is excluded: it belongs to its decode budget.
    prescaled = [detector._prescale(frame)[0] for frame in frames]
    weight_digest = hashlib.sha256()
    with a.weights.open("rb") as weight_file:
        for block in iter(lambda: weight_file.read(1024 * 1024), b""):
            weight_digest.update(block)
    weight_hash = weight_digest.hexdigest()
    report = {"gpu": gpu, "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "visible_count": torch.cuda.device_count(), "platform": platform.platform(),
              "gpu_total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
              "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "torch_cpu_threads": torch.get_num_threads(),
              "ultralytics": version("ultralytics"), "opencv": cv2.__version__,
              "weights_sha256": weight_hash,
              "video": str(a.video), "source_size": a.video.stat().st_size,
              "frame_indices": indices, "source_shape": list(frames[0].shape),
              "duration_sec": n / fps, "cpu_decode_preload_sec": decode_sec,
              "config": {"imgsz": 960, "conf": 0.1, "iou": 0.6, "half": True,
                         "coco_classes": detector.coco_ids,
                         "batch": 1, "prescale_width": 1920, "warmup": a.warmup,
                         "repeats": a.repeats}, "passes": {}}
    for name, inputs in (("part_a_1920_bgr", prescaled), ("part_b_native_bgr", frames)):
        for i in range(a.warmup):
            detector.predict([inputs[i % len(inputs)]])
        torch.cuda.synchronize()
        if not detector.model.predictor.model.fp16:
            raise RuntimeError("Predictor did not enable FP16")
        torch.cuda.reset_peak_memory_stats()
        elapsed, counts = [], []
        for _ in range(a.repeats):
            for frame in inputs:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                detections = detector.predict([frame])[0]
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - t0)
                counts.append(len(detections.xyxy))
        report["passes"][name] = {
            "calls": len(elapsed), "total_sec": sum(elapsed),
            "fps": len(elapsed) / sum(elapsed), "mean_ms": float(np.mean(elapsed) * 1000),
            "p50_ms": float(np.percentile(elapsed, 50) * 1000),
            "p95_ms": float(np.percentile(elapsed, 95) * 1000),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "latencies_sec": elapsed, "detection_counts": counts}
        print(json.dumps({name: report["passes"][name]}, indent=2), flush=True)
    fa = report["passes"]["part_a_1920_bgr"]["fps"]
    fb = report["passes"]["part_b_native_bgr"]["fps"]
    report["detection_only_budget"] = [{"stride_a": 3, "stride_b": s,
        "seconds": ((n + 2) // 3) / fa + ((n + s - 1) // s) / fb}
        for s in (3, 5, 6)]
    report["limitations"] = "Real-frame sampled detector throughput, not full submission. Add both CPU decode passes, tracking, rules, signal/other models and serialization; verify full harness on T4 host."
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
