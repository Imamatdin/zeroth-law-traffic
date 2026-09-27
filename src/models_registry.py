"""Load one explicit local checkpoint on import; give every consumer a private predictor."""
import copy
import json
import os
import random
from pathlib import Path
# Set before importing CUDA libraries; seed every RNG before model construction.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import numpy as np
from src.perception.detector import UltralyticsDetector

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

# Missing optional dependencies must fail, never invoke pip during judging.
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/perception.json").read_text(encoding="utf-8"))
WEIGHTS = (ROOT / CONFIG["detector"]["weights"]).resolve()
if not WEIGHTS.is_file() or WEIGHTS.suffix != ".pt":
    raise FileNotFoundError(f"Required local checkpoint missing: {WEIGHTS}; automatic downloads disabled")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(8, torch.get_num_threads()))
_LOADED = UltralyticsDetector(**{**CONFIG["detector"], "weights": WEIGHTS,
                               "device": DEVICE, "half": DEVICE != "cpu"})


def _new_wrapper():
    # YOLO predictor holds mutable call state. Its backend uses the loaded module,
    # never the checkpoint filename, and constructs private inference state.
    detector = copy.copy(_LOADED)
    detector.model = copy.copy(_LOADED.model)
    detector.model.overrides = dict(_LOADED.model.overrides)
    detector.model.callbacks = copy.deepcopy(_LOADED.model.callbacks)
    detector.model.predictor = None
    return detector


_PREDICTOR = _LOADED.model._smart_load("predictor")
_OVERRIDES = {**_LOADED.model.overrides, "mode": "predict", "task": "detect", "save": False,
              "verbose": False, "imgsz": _LOADED.imgsz, "device": DEVICE,
              "quantize": 16 if DEVICE != "cpu" else None}
_BACKENDS = {}
for _part in ("A", "B"):
    _predictor = _PREDICTOR(overrides=dict(_OVERRIDES))
    _predictor.setup_model(_LOADED.model.model, verbose=False)
    _predictor.model.warmup(imgsz=(1, 3, 960, 960))
    _BACKENDS[_part] = _predictor.model
del _predictor


def new_detector(part="A"):
    detector = _new_wrapper()
    predictor = _PREDICTOR(overrides=dict(_OVERRIDES))
    predictor.model = _BACKENDS[part]
    predictor.device = predictor.model.device
    predictor.done_warmup = True
    detector.model.predictor = predictor
    return detector


# Exercise preprocessing, model forward and NMS for each real input shape at import.
# These are synthetic warm-up frames, never footage or Part A observations.
for _part, _shape in (("A", (1080, 1920, 3)), ("B", (2160, 3840, 3))):
    _warm = new_detector(_part)
    for _ in range(3):
        _warm.predict([np.zeros(_shape, dtype=np.uint8)])
del _warm
