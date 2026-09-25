"""Build/reuse a real detector+tracker cache through a Stage 3 pipeline factory."""

import argparse
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.cache import build_cache


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--factory", required=True, help="module:function accepting (meta, config)")
    p.add_argument("--config", required=True, type=Path, help="JSON detector/tracker parameters")
    p.add_argument("--weights", required=True, action="append", type=Path,
                   help="Every weight file used; included in cache identity")
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()
    config = json.loads(a.config.read_text(encoding="utf-8"))
    module, name = a.factory.split(":", 1)
    if not module.startswith("src.perception."):
        p.error("Pipeline factories must live under src.perception so their code is fingerprinted")
    packages = {}
    for package in ("numpy", "opencv-python", "opencv-python-headless", "pyarrow",
                    "torch", "torchvision", "ultralytics", "scipy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    # Event/world-model changes must not invalidate expensive perception caches.
    code = [*(ROOT / "src" / "perception").rglob("*.py"), ROOT / "src" / "contracts.py"]
    if (ROOT / "src" / "models_registry.py").exists():
        code.append(ROOT / "src" / "models_registry.py")
    pipeline = {"factory": a.factory, "config": config,
                "weights": [{"file": w.name, "sha256": sha256(w)} for w in a.weights],
                "python": sys.version, "packages": packages,
                "code_sha256": {str(f.relative_to(ROOT)): sha256(f) for f in sorted(code)}}

    def factory(meta):
        return getattr(importlib.import_module(module), name)(meta, config)

    meta = build_cache(a.video, a.out, pipeline, factory, a.stride)
    print(json.dumps({"cache": str(a.out), "row_counts": meta["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
