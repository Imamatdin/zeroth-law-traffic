"""Measure the full demo pipeline (CPU decode, A and causal B), no cache use."""
import argparse
import json
import platform
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from demo.engine import Engine
    start = time.perf_counter()
    engine = Engine()
    setup = time.perf_counter() - start
    records = []
    args.out.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        def progress(stage, fraction):
            bucket = int(fraction * 4)
            key = (stage, bucket)
            if key != progress.last:
                print(video.name, stage, f'{fraction:.0%}', flush=True)
                progress.last = key
        progress.last = None
        result = engine.process(video, progress)
        (args.out / (video.stem + '.json')).write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
        record = dict(video=str(video), duration=result['replay']['duration'],
                      width=result['replay']['width'], height=result['replay']['height'],
                      track_count=len(result['replay']['tracks']), detections=len(result['detections']['rows']),
                      **result['metadata'])
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.out / 'timing.json').write_text(json.dumps(dict(platform=platform.platform(),
        setup_seconds=setup, runs=records), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
