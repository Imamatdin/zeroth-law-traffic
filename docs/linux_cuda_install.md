# Linux judging environment

For Linux x86_64, Python 3.11 and glibc 2.28 or newer:

```sh
python -m pip install --require-hashes -r requirements-linux-cu126.lock
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The lock resolves the complete runtime dependency graph with binary wheels for
that target. It pins `torch==2.10.0+cu126`, `torchvision==0.25.0+cu126`, and
`ultralytics-opencv-headless==8.4.163`. Explicit CUDA build suffixes prevent pip
from substituting a default CUDA 13 build. Only headless OpenCV is selected.
Cache tooling's extra dependencies remain in `requirements-dev.txt`.

PyTorch lists 2.10.0 / torchvision 0.25.0 as a supported CUDA 12.6 pair:
https://pytorch.org/get-started/previous-versions/

NVIDIA documents CUDA 12.x minor-version compatibility starting with Linux
driver 525.60.13. This has feature/PTX limitations; dependency resolution is not
a hardware compatibility test:
https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

Run `scripts/t4_end_to_end.py` on the actual T4 environment to verify CUDA
execution, the full-pipeline wall-time ratio and guard stride. The Windows CPU
test suite does not establish a T4 runtime or validate this Linux installation.
CUDA 12.6 is the judging/T4 target, not the project's RTX 5060 development target.

Regenerate the lock with the exact `uv pip compile` command recorded in its header.
