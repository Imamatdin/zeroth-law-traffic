#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 && $# -le 2 ]] || { echo "Usage: $0 FINAL_TAG [SAMPLE_DIRECTORY]" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
videos=$(realpath -- "${2:-$root/data/samples}")
cd "$root"
tag_commit=$(git rev-parse --verify "refs/tags/$1^{commit}")
[[ $(git rev-parse HEAD) == "$tag_commit" ]] || { echo 'Checkout the supplied final tag first' >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Tracked files must be clean' >&2; exit 2; }
[[ ! -e predictions_samples.json ]] || { echo 'predictions_samples.json already exists; preserve it before rerunning' >&2; exit 2; }
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
py=${PYTHON:-python}
"$py" - "$videos" <<'PY'
import pathlib, sys, torch
if not torch.cuda.is_available():
    raise SystemExit('Final sample generation requires a CUDA GPU')
expected = {'C3896.MP4', 'C3897.MP4', 'C3905.MP4'}
present = {p.name for p in pathlib.Path(sys.argv[1]).iterdir() if p.is_file()}
if expected - present:
    raise SystemExit(f'Missing provided samples: {sorted(expected - present)}')
print('GPU:', torch.cuda.get_device_name(0))
PY
bash weights/download.sh
mkdir -p outputs
candidate=$(mktemp "$root/outputs/predictions_samples.XXXXXX.json")
# No no-risk flag, external stride, or relaxed time budget: use the official defaults.
"$py" run_submission.py --videos "$videos" --out "$candidate" --team zeroth-law
"$py" evaluate.py --pred "$candidate" --validate-only
"$py" scripts/check_harness_output.py --pred "$candidate" --videos "$videos"
mv -- "$candidate" predictions_samples.json
printf 'Generated predictions_samples.json from %s (%s)\n' "$1" "$tag_commit"
