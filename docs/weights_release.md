# Publish weights-v1

Published: https://github.com/Imamatdin/zeroth-law-traffic/releases/tag/weights-v1 (download and SHA256 check verified 2026-09-27). The steps used, kept for re-publication:

1. Open https://github.com/Imamatdin/zeroth-law-traffic/releases/new while signed in.
2. Create tag **weights-v1**, target the pushed **eng/package** branch; title **weights-v1**.
3. Upload `C:\Users\imama\Projects\zeroth-law-traffic\weights\yolo11m.pt` (40,684,120 bytes), `weights/SHA256SUMS` and `third_party/Ultralytics-AGPL-3.0.txt` from eng/package.
4. Paste docs/weights_release_notes.md as the description.
5. Publish (not draft/prerelease). Keep asset name exactly `yolo11m.pt`.
6. On a clean machine run `bash weights/download.sh`. Expected: `Verified shipped weights: 40684120 bytes`. Repeat to verify the already-present path, then run the clean-install script.

If gh becomes available, from this checkout with the checkpoint present:

```bash
gh release create weights-v1 weights/yolo11m.pt weights/SHA256SUMS \
  third_party/Ultralytics-AGPL-3.0.txt --target eng/package \
  --title weights-v1 --notes-file docs/weights_release_notes.md
```

Check any existing release against the digest before retrying; do not overwrite assets silently.
