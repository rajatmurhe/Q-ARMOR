import unittest
import numpy as np
from tools.calibrate_game import fit_candidate
from tools.data_pipeline import validation_partition


class CalibrationTests(unittest.TestCase):
    def test_partition_is_disjoint_stratified_and_repeatable(self):
        labels = np.repeat(['a', 'b', 'c'], 20)
        a, b = validation_partition(labels)
        c, d = validation_partition(labels)
        self.assertEqual(set(a) & set(b), set())
        self.assertEqual(set(a) | set(b), set(range(60)))
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertEqual(set(labels[a]), set(labels[b]))

    def test_candidate_uses_signed_effects_and_ignores_audit(self):
        def metric(f1):
            return {'per_class': {'attack': {'f1': f1}}, 'false_alarm_rate': .2}
        records = [dict(requested_scale=1.15, target='attack', before=metric(.5),
                        control=metric(.6), reweighted=metric(after),
                        audit={'irrelevant': 999}) for after in [.7, .1]]
        result = fit_candidate(records, 1.15)
        self.assertAlmostEqual(result['effectiveness']['NORMAL'][0], .2)
        # Signed reductions .4 and -.8 average negative: no artificial benefit.
        self.assertEqual(result['effectiveness']['CLASS_REWEIGHTING'], [0., 0.])
        self.assertEqual(result['false_positive_cost']['NORMAL'], .2)


if __name__ == '__main__':
    unittest.main()
