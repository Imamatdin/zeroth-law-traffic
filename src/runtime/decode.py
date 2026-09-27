"""CPU sampled decode, source indices, coordinate restoration and OpenCV fallback."""
from pathlib import Path
import subprocess
import tempfile
import cv2
import numpy as np
from src.contracts import Detections
from src.runtime.budget import log


def metadata(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {path}")
        out = dict(video_id=Path(path).name, fps=cap.get(cv2.CAP_PROP_FPS),
                   width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if any(out[k] <= 0 for k in ("fps", "width", "height", "n_frames")):
            raise ValueError(f"Invalid video metadata: {out}")
        return out
    finally:
        cap.release()


def scaled_size(meta):
    width = min(1920, meta["width"])
    return width, max(2, round(meta["height"] * width / meta["width"] / 2) * 2)


def restore(dets, shape, meta):
    xyxy = dets.xyxy.copy()
    xyxy[:, [0, 2]] *= meta["width"] / shape[1]
    xyxy[:, [1, 3]] *= meta["height"] / shape[0]
    return Detections(xyxy, dets.cls, dets.score)


def _read_frame(pipe, size):
    data = bytearray()
    while len(data) < size:
        chunk = pipe.read(size - len(data))
        if not chunk:
            if data:
                raise OSError(f"Truncated raw frame: {len(data)}/{size}")
            return None
        data.extend(chunk)
    return data


def opencv_frames(path, meta, start=0):
    cap = cv2.VideoCapture(str(path))
    size = scaled_size(meta)
    try:
        if not cap.isOpened():
            raise OSError(f"OpenCV fallback cannot open {path}")
        for idx in range(meta["n_frames"]):
            if not cap.grab():
                raise OSError(f"OpenCV ended at frame {idx}/{meta['n_frames']}")
            if idx >= start and idx % 3 == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    raise OSError(f"OpenCV retrieve failed at {idx}")
                yield idx, cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    finally:
        cap.release()


def frames(path, meta, *, force_opencv=False):
    if force_opencv:
        log("decode", backend="opencv", cpu=True)
        yield from opencv_frames(path, meta)
        return
    next_idx, proc, failure = 0, None, None
    width, height = scaled_size(meta)
    with tempfile.TemporaryFile() as stderr:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()  # Wheel binary; never downloads.
            vf = (f"select=not(mod(n\\,3)),scale={width}:{height}:flags=bicubic:"
                  "in_color_matrix=bt601:in_h_chr_pos=128")
            cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-hwaccel", "none",
                   "-threads", "8", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
                   "-filter_threads", "8", "-vf", vf, "-fps_mode", "passthrough", "-pix_fmt", "bgr24",
                   "-c:v", "rawvideo", "-threads:v", "1", "-f", "rawvideo", "pipe:1"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            log("decode", backend="ffmpeg", cpu=True, command=cmd)
            while True:
                data = _read_frame(proc.stdout, width * height * 3)
                if data is None:
                    break
                if next_idx >= meta["n_frames"]:
                    raise OSError("More frames than container metadata")
                idx, next_idx = next_idx, next_idx + 3
                yield idx, np.frombuffer(data, np.uint8).reshape(height, width, 3)
            code = proc.wait(timeout=30)
            if code or next_idx < meta["n_frames"]:
                raise OSError(f"FFmpeg exit={code}, next source frame={next_idx}")
        except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            failure = str(exc)
        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                proc.stdout.close()
        if failure is not None:
            stderr.seek(0)
            log("decode_fallback", reason=failure, stderr=stderr.read(4000).decode(errors="replace"),
                resume_frame=next_idx, quality="OpenCV INTER_AREA differs from FFmpeg bicubic")
            yield from opencv_frames(path, meta, start=next_idx)
