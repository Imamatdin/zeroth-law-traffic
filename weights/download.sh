#!/usr/bin/env bash
set -euo pipefail
weights_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$weights_dir"
command -v curl >/dev/null || { echo 'curl is required' >&2; exit 1; }
command -v sha256sum >/dev/null || { echo 'sha256sum is required' >&2; exit 1; }
base='https://github.com/Imamatdin/zeroth-law-traffic/releases/download/weights-v1'
tmp=''
trap 'if [[ -n "$tmp" ]]; then rm -f -- "$tmp"; fi' EXIT
while read -r digest name; do
  [[ -n "$digest" ]] || continue
  [[ "$name" != */* && "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo 'Invalid manifest' >&2; exit 1; }
  if [[ -f "$name" ]] && printf '%s  %s\n' "$digest" "$name" | sha256sum --check --status; then
    echo "$name: verified (already present)"
    continue
  fi
  tmp=$(mktemp "./.${name}.download.XXXXXX")
  curl --fail --location --retry 3 --connect-timeout 30 --proto '=https' --proto-redir '=https' \
    --output "$tmp" "$base/$name"
  printf '%s  %s\n' "$digest" "$tmp" | sha256sum --check --status || {
    echo "SHA256 mismatch for $name; existing file left untouched" >&2; exit 1;
  }
  mv -f -- "$tmp" "$name"
  tmp=''
  echo "$name: downloaded and verified"
done < SHA256SUMS
# Only manifest-listed weights are shipped; fail if their total exceeds the spec.
total=0
while read -r digest name; do
  [[ -n "$digest" ]] || continue
  bytes=$(wc -c < "$name")
  total=$((total + bytes))
done < SHA256SUMS
(( total <= 5000000000 )) || { echo 'Shipped weights exceed 5 GB' >&2; exit 1; }
echo "Verified shipped weights: $total bytes"
