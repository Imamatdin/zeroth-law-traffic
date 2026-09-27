"""Bounded CPU decode for untrusted uploads."""
from pathlib import Path
import math
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np
from src.runtime.decode import _read_frame

MAX_SECONDS = 120.0
MAX_BYTES = 3 * 1024**3


class InvalidVideo(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def inspect_video(path):
    with open(path, 'rb') as f:
        header = f.read(32)
    if len(header) < 12 or header[4:8] != b'ftyp' or header[8:12] == b'qt  ':
        raise InvalidVideo('wrong_format', 'Upload an MP4 video, not a renamed image or other file.')
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(fps) or not 1 <= fps <= 120 or n < 1 or min(w, h) < 16:
            raise InvalidVideo('invalid_video', 'Cannot read valid video metadata.')
        if w > 4096 or h > 4096 or w * h > 4096 * 2160:
            raise InvalidVideo('resolution_limit', 'Maximum supported resolution is 4096 x 2160 pixels (either orientation).')
        duration = n / fps
        if duration > MAX_SECONDS:
            raise InvalidVideo('too_long', 'Video must be at most 120 seconds.')
        ok, first = cap.read()
        if not ok:
            raise InvalidVideo('invalid_video', 'Cannot decode the first frame.')
        return dict(video_id=Path(path).name, fps=fps, width=w, height=h,
                    n_frames=n, duration=duration), first
    finally:
        cap.release()


def sampled_frames(path, meta, stride, threads=2, width=960):
    width = min(width, meta['width'])
    height = max(2, round(meta['height'] * width / meta['width'] / 2) * 2)
    vf = (f'select=not(mod(n\\,{stride})),scale={width}:{height}:flags=bicubic:'
          'in_color_matrix=bt601:in_h_chr_pos=128')
    with tempfile.TemporaryFile() as err:
        process = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-xerror', '-nostdin',
            '-hwaccel', 'none', '-threads', str(threads), '-i', str(path), '-map', '0:v:0',
            '-an', '-sn', '-dn', '-t', str(MAX_SECONDS + 1), '-filter_threads', str(threads),
            '-vf', vf, '-fps_mode', 'passthrough', '-pix_fmt', 'bgr24', '-c:v', 'rawvideo',
            '-threads:v', '1', '-f', 'rawvideo', 'pipe:1'], stdout=subprocess.PIPE, stderr=err)
        count = 0
        try:
            while True:
                buf = _read_frame(process.stdout, width * height * 3)
                if buf is None:
                    break
                idx = count * stride
                if idx / meta['fps'] >= MAX_SECONDS:
                    raise InvalidVideo('too_long', 'Decoded video exceeds 120 seconds.')
                count += 1
                yield idx, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            code = process.wait(timeout=30)
            if code or count != math.ceil(meta['n_frames'] / stride):
                raise InvalidVideo('decode_failed', 'Video is truncated, corrupt, or has inconsistent frame metadata.')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=30)
            process.stdout.close()
