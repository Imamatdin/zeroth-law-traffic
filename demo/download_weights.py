"""Build-time release download; never called by the running application."""
import hashlib
from pathlib import Path
import urllib.error
import urllib.request

URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
SHA256 = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'


def main():
    target = Path('weights/yolo11n.pt')
    target.parent.mkdir(exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        return
    partial = target.with_suffix('.partial')
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(URL, timeout=120) as source, partial.open('wb') as out:
            while chunk := source.read(1024**2):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != SHA256:
            raise RuntimeError('Ultralytics v8.3.0/yolo11n.pt checksum mismatch')
        partial.replace(target)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Ultralytics v8.3.0/yolo11n.pt unavailable (HTTP {exc.code}). The pinned upstream asset must be accessible to the Space builder.') from exc
    finally:
        partial.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
