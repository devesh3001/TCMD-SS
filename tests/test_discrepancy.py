import unittest
import torch
from src.generative.discrepancy import discrepancy_channels, fit_robust, clean_threshold, fused_score
from src.diffusion.process import covering_masks


class DiscrepancyTests(unittest.TestCase):
    def test_identity_has_no_discrepancy(self):
        torch.manual_seed(42)
        x = torch.rand(2, 1, 128, 128)
        tokens = torch.randn(2, 16, 8)
        scores, maps = discrepancy_channels(x, x, torch.zeros_like(x), tokens, tokens)
        self.assertTrue(torch.all(scores.abs() < 1e-5))
        self.assertEqual(tuple(maps.shape), (2, 4, 128, 128))

    def test_four_complementary_tile_masks(self):
        masks = covering_masks(128)
        self.assertTrue(torch.equal(masks.sum(0), torch.ones(1,128,128)))
        self.assertEqual(float(masks[0].sum()), 4096)

    def test_clean_calibration_and_continuity(self):
        a = torch.arange(40.).reshape(10,4)
        median, scale = fit_robust(a, clean_only=True)
        with self.assertRaises(ValueError):
            fit_robust(a, clean_only=False)
        threshold = clean_threshold(a+1, median, scale, clean_only=True)
        self.assertTrue(torch.isfinite(threshold))
        values = fused_score(torch.stack([a[-1]*10,a[-1]*100]), median, scale)
        self.assertGreater(float(values[1]), float(values[0]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
