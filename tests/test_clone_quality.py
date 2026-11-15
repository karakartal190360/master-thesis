import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_clone_quality import error_rates, normalize  # noqa: E402


class ErrorRateTest(unittest.TestCase):
    def test_case_and_punctuation_ignored(self):
        self.assertEqual(error_rates("Die Fischer, müde!", "die fischer müde"), (0.0, 0.0))

    def test_substitution_and_deletion(self):
        self.assertEqual(error_rates("a b c d", "a x c"), (0.5, 0.5))

    def test_umlaut_kept(self):
        self.assertEqual(normalize("Für Bürger."), "für bürger")


if __name__ == "__main__":
    unittest.main()
