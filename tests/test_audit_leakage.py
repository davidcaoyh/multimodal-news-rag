import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.audit_leakage import dhash64, hamming64, normalize_text


class LeakageAuditTests(unittest.TestCase):
    def test_normalize_text_ignores_case_and_punctuation(self):
        self.assertEqual(normalize_text("Hello, WORLD!"), "hello world")

    def test_dhash_is_stable_under_lossless_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.png"
            b = Path(tmp) / "b.png"
            image = Image.new("L", (90, 80))
            image.putdata([(x * 3 + y * 5) % 256 for y in range(80) for x in range(90)])
            image.save(a)
            image.resize((180, 160)).save(b)
            self.assertLessEqual(hamming64(dhash64(a), dhash64(b)), 2)

    def test_hamming64(self):
        self.assertEqual(hamming64(0b1010, 0b0011), 2)


if __name__ == "__main__":
    unittest.main()

