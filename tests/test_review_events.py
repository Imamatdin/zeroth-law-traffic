import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.review_events import find_caches, summarise


def write_cache(root: Path, name: str, complete: bool) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"video_id": f"{name}.MP4", "complete": complete}), encoding="utf-8")
    return d


class ReviewEventsTests(unittest.TestCase):
    def test_caches_are_found_zips_unpacked_and_incomplete_ones_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            write_cache(root, "A", True)
            write_cache(root, "B", False)
            src = write_cache(Path(tmp) / "elsewhere", "C", True)
            with zipfile.ZipFile(root / "C.zip", "w") as z:
                z.write(src / "meta.json", "C/meta.json")
            found = [p.name for p in find_caches(None, root)]
            self.assertEqual(sorted(found), ["A", "C"])

    def test_summary_counts_verdicts_and_only_calls_fully_reviewed_classes_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = [{"label": "jaywalking", "verdict": "yes"}, {"label": "jaywalking", "verdict": "no"},
                    {"label": "near_miss", "verdict": None}, {"label": "stop_line", "verdict": "yes"}]
            (out / "V1").mkdir()
            (out / "V1" / "detections.json").write_text(json.dumps(rows), encoding="utf-8")
            summarise(out, {"V1": {"classes_run": ["jaywalking", "near_miss", "stop_line", "accident"]}})
            table = json.loads((out / "summary.json").read_text(encoding="utf-8"))["classes"]
            self.assertEqual(table["jaywalking"]["V1"]["status"], "1 false fire")
            self.assertEqual(table["near_miss"]["V1"]["status"], "1 pending review")
            self.assertEqual(table["stop_line"]["V1"]["status"], "no false fires")
            self.assertEqual(table["accident"]["V1"]["status"], "no false fires")
            self.assertEqual(table["fire_smoke"]["V1"]["status"], "not run (no video)")


if __name__ == "__main__":
    unittest.main()
