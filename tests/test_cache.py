"""Cache storage/invalidation unit tests. No fabricated video or model claims."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pyarrow.parquet as pq

from src.contracts import Detections
from src.perception.cache import CacheWriter, build_cache, read_meta


class CacheTests(unittest.TestCase):
    def test_reuse_never_constructs_models_or_opens_video(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            identity = {"path": "not-opened", "size": 1, "mtime_ns": 1}
            writer = CacheWriter(directory)
            writer.finish({"request": {"source": identity, "pipeline": {"model": "test"}, "stride": 2}})
            factory = Mock(side_effect=AssertionError("Models must not be loaded"))
            with patch("src.perception.cache.source_identity", return_value=identity), \
                 patch("src.perception.cache.cv2.VideoCapture") as decoder:
                build_cache(Path("not-opened"), directory, {"model": "test"}, factory, 2)
            factory.assert_not_called()
            decoder.assert_not_called()

    def test_empty_detections_preserve_sampled_frame_and_schemas(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            writer = CacheWriter(path)
            writer.append(30, 1.001, Detections.empty(), [])
            writer.finish({"request": {"stride": 30}})
            meta = read_meta(path, {"stride": 30})
            self.assertEqual(meta["row_counts"], {"frames": 1, "detections": 0, "tracks": 0})
            self.assertEqual(pq.read_table(path / "frames.parquet").to_pylist(), [{"frame": 30, "t": 1.001}])

    def test_stale_and_incomplete_caches_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            writer = CacheWriter(path)
            writer.close()
            with self.assertRaises(FileNotFoundError):
                read_meta(path)
            writer = CacheWriter(path)
            writer.finish({"request": {"weights": "old"}})
            with self.assertRaisesRegex(ValueError, "Stale cache"):
                read_meta(path, {"weights": "new"})

    def test_manifest_table_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            writer = CacheWriter(path)
            writer.finish({"request": {}})
            manifest = json.loads((path / "meta.json").read_text())
            manifest["row_counts"]["tracks"] = 1
            (path / "meta.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "schema/count mismatch"):
                read_meta(path)


if __name__ == "__main__":
    unittest.main()
