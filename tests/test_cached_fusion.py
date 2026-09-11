import json
import unittest
from scripts.evaluate_cached_fusion import ROOT, read, metrics, quantile


class CachedMetricsTests(unittest.TestCase):
    def test_matches_saved_sklearn_metrics(self):
        rows=read(ROOT/'outputs/evaluation/predictions.csv')
        reference=json.loads((ROOT/'outputs/evaluation/metrics.json').read_text())
        actual=metrics(rows,[float(r['score']) for r in rows],reference['threshold'])
        for key in ['auroc','auprc','precision','recall','f1']:
            self.assertAlmostEqual(actual[key],reference[key],places=12)
        self.assertAlmostEqual(actual['fpr'],reference['false_positive_rate'],places=12)

    def test_ties_get_half_auc_credit(self):
        rows=[{'base_id':'a','is_anomaly':0},{'base_id':'a','is_anomaly':1}]
        result=metrics(rows,[1.,1.],1.)
        self.assertEqual(result['auroc'],.5)
        self.assertEqual(result['auprc'],.5)
        self.assertEqual(result['paired_anomaly_gt_clean'],0.)

    def test_interpolated_quantile(self):
        self.assertAlmostEqual(quantile([0.,1.,2.,3.],.95),2.85)


if __name__=='__main__':
    unittest.main()
