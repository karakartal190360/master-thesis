import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_emime import holm, margin_reading, sign_flip  # noqa: E402


class SignFlipTest(unittest.TestCase):
    def test_all_positive_gives_smallest_p(self):
        # only the observed pattern and its mirror reach |mean|: 2 / 2^14
        self.assertAlmostEqual(sign_flip([1.0] * 14), 2 / 2 ** 14)

    def test_zero_effect_gives_p_one(self):
        self.assertEqual(sign_flip([0.0] * 14), 1.0)


class HolmTest(unittest.TestCase):
    def test_step_down_and_monotone(self):
        self.assertEqual(holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])


class MarginTest(unittest.TestCase):
    def test_readings(self):
        self.assertTrue(margin_reading([-1.0, 1.5]).startswith("no meaningful change"))
        self.assertEqual(margin_reading([2.5, 6.0]), "meaningful loss")
        self.assertEqual(margin_reading([0.5, 4.0]), "loss, size uncertain")
        self.assertEqual(margin_reading([-0.5, 4.0]), "inconclusive")


if __name__ == "__main__":
    unittest.main()
