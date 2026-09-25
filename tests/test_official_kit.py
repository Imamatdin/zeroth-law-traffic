import hashlib
from pathlib import Path
import unittest


class OfficialKitTests(unittest.TestCase):
    def test_harness_and_evaluator_stay_byte_identical(self):
        expected = {
            "run_submission.py": "a47b494afae14432a65166b43cd5f2a278408ce661aecf0fb86b6fa05a06c204",
            "evaluate.py": "111c6fa04709c9f1df4ea3db4bede749953b2c27bdc5679c2ef56794637573b5",
        }
        root = Path(__file__).resolve().parents[1]
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
