import numpy as np
import unittest
import os
from recommender_pkg.metrics import dcg_score
from recommender_pkg.metrics import ndcg_score
from recommender_pkg.metrics import perform_groupwise_evaluation


class TestMetrics(unittest.TestCase):
    def test_dcg_score(self):
        self.assertAlmostEqual(dcg_score([1, 2, 3, 4]), 5.4846, places=4)
        self.assertAlmostEqual(dcg_score([4, 3, 2, 1]), 7.3235, places=4)

    def test_ndcg_score(self):
        self.assertAlmostEqual(ndcg_score([4, 3, 2, 1]), 1)
        self.assertAlmostEqual(ndcg_score([1, 2, 3, 4]), 0.7489, places=4)

    def test_perform_groupwise_evaluation(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        X_test = np.load(os.path.join(base_dir, "data/X_test_implicit.npy"))
        y_test = np.load(os.path.join(base_dir, "data/y_test_implicit.npy"))
        y_pred = np.load(os.path.join(base_dir, "data/y_pred_implicit.npy"))
        expected = {"hit_ratio": 0.5,
                    "normalized_discounted_cumulative_gain": 0.25}

        self.assertEqual(perform_groupwise_evaluation(X_test, y_test, y_pred),
                         expected)


if __name__ == '__main__':
    unittest.main()
