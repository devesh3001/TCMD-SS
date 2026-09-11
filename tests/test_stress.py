import unittest
import torch
from src.generative.stress import FAMILIES, stress, pil_image, tensor_image
from scripts.run_generative_development import score_maps


class StressTests(unittest.TestCase):
    def test_six_families_have_reproducible_visible_effects(self):
        torch.manual_seed(42)
        image=pil_image(torch.rand(1,128,128))
        for family in FAMILIES:
            for severity in [.15,.30,.50]:
                with self.subTest(family=family,severity=severity):
                    changed,mask=stress(image,family,severity,42)
                    again,again_mask=stress(image,family,severity,42)
                    self.assertEqual(changed.tobytes(),again.tobytes())
                    self.assertEqual(mask.tobytes(),again_mask.tobytes())
                    self.assertNotEqual(changed.tobytes(),image.tobytes())
                    outside=tensor_image(mask)==0
                    self.assertTrue(torch.equal(tensor_image(changed)[outside],tensor_image(image)[outside]))

    def test_identical_reconstruction_has_zero_scores(self):
        x=torch.rand(2,1,128,128)
        scores,maps=score_maps(x,x,torch.zeros_like(x))
        self.assertTrue(torch.equal(scores,torch.zeros(2,3)))
        self.assertEqual(tuple(maps.shape),(2,3,128,128))


if __name__=='__main__':unittest.main()
