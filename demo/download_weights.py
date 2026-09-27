"""Build-time release download; never called by the running application."""
import hashlib
from pathlib import Path
import urllib.error
import urllib.request

URL = 'https://github.com/Imamatdin/zeroth-law-traffic/releases/download/weights-v1/yolo11m.pt'
SHA256 = 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95'


def main():
    target = Path('weights/yolo11m.pt')
    target.parent.mkdir(exist_ok=True)
    partial = target.with_suffix('.partial')
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(URL, timeout=120) as source, partial.open('wb') as out:
            while chunk := source.read(1024**2):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != SHA256:
            raise RuntimeError('weights-v1/yolo11m.pt checksum mismatch')
        partial.replace(target)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'weights-v1/yolo11m.pt unavailable (HTTP {exc.code}). Publish the public release asset before building the Space.') from exc
    finally:
        partial.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
