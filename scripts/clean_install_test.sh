#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: $0 /path/to/sample.mp4" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
clip=$(realpath -- "$1")
[[ -f "$clip" ]] || { echo "Missing input: $clip" >&2; exit 2; }
cd "$root"
[[ $(uname -s) == Linux ]] || { echo 'Run on the Linux judging/GPU machine' >&2; exit 2; }
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
run_dir=$(mktemp -d "$root/outputs-clean-install.XXXXXX")
echo "Logs, fresh environment and output retained at $run_dir"
exec > >(tee "$run_dir/run.log") 2>&1
"${PYTHON:-python3.11}" -m venv "$run_dir/venv"
py="$run_dir/venv/bin/python"
"$py" -m pip install -r requirements.txt
"$py" -m pip check
"$py" -m pip freeze > "$run_dir/installed.txt"
bash weights/download.sh
mkdir "$run_dir/videos"
# Stream-copy the first 20 seconds; retain the source codec/pixel format.
"$py" - "$clip" "$run_dir/videos/clip20.mp4" <<'PY'
import subprocess, sys, imageio_ffmpeg
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
                '-i', sys.argv[1], '-t', '20', '-map', '0:v:0', '-an',
                '-c:v', 'copy', sys.argv[2]], check=True)
PY
"$py" run_submission.py --videos "$run_dir/videos" --out "$run_dir/predictions.json" --team zeroth-law
"$py" evaluate.py --pred "$run_dir/predictions.json" --validate-only
"$py" scripts/check_harness_output.py --pred "$run_dir/predictions.json" --videos "$run_dir/videos"
echo 'Clean install and official 3x harness check passed'
