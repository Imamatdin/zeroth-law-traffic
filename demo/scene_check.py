from pathlib import Path

import cv2
import numpy as np

REFERENCE = Path(__file__).parent / 'assets/reference.jpg'


def check_scene(frame, width, height, reference=REFERENCE):
    """Conservative static-camera gate, not semantic scene recognition."""
    reasons = []
    if (width, height) != (3840, 2160):
        reasons.append('resolution_mismatch')
    if abs(width / height - 16 / 9) > .01:
        reasons.append('aspect_mismatch')
    ref = cv2.imread(str(reference))
    if ref is None:
        raise RuntimeError('Reference camera image is missing')
    def gray(im):
        return cv2.cvtColor(cv2.resize(im, (480, 270)), cv2.COLOR_BGR2GRAY)
    a, b = gray(frame), gray(ref)
    # Blurred edges reduce sensitivity to moving traffic and exposure.
    def edges(im):
        return cv2.GaussianBlur(cv2.Canny(im, 60, 150).astype(np.float32), (9, 9), 0)
    x, y = edges(a).ravel(), edges(b).ravel()
    x, y = x - x.mean(), y - y.mean()
    similarity = float(x @ y / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-9))
    if similarity < .55:
        reasons.append('low_reference_similarity')
    return {'matches': not reasons, 'reasons': reasons, 'similarity': round(similarity, 4),
            'threshold': .55, 'reference_resolution': [3840, 2160],
            'method': 'blurred-edge normalized correlation; first decoded frame',
            'note': 'Conservative camera check; not a calibrated scene classifier.'}
