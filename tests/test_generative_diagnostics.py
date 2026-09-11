import unittest
import torch
from scripts.refine_generative_scores import robust_scale
from scripts.evaluate_generative_localization import pixel_auc
from scripts.evaluate_rare_prevalence import prevalence_metrics


class DiagnosticTests(unittest.TestCase):
    def test_zero_mad_uses_normal_spread(self):
        values=torch.tensor([[0.],[0.],[0.],[0.],[1.],[2.]])
        median,scale=robust_scale(values)
        self.assertEqual(float(median),0.)
        self.assertGreater(float(scale),.1)

    def test_pixel_auc_handles_ties_and_missing_background(self):
        self.assertEqual(pixel_auc(torch.tensor([0,1]),torch.tensor([0.,1.])),1.)
        self.assertEqual(pixel_auc(torch.tensor([0,1]),torch.tensor([1.,1.])),.5)
        self.assertIsNone(pixel_auc(torch.tensor([1,1]),torch.tensor([0.,1.])))

    def test_rare_prevalence_precision_matches_bayes(self):
        result=prevalence_metrics([0,0,1,1],[0.,1.,1.,1.],.5,.05)
        self.assertAlmostEqual(result['expected_precision'],.05/(.05+.95*.5))


if __name__=='__main__':unittest.main()
