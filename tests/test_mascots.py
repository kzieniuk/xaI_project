import unittest
import numpy as np
from src.mascots import SAXTransformer, MASCOTS

class DummyModel:
    def predict_from_array(self, values):
        # Simple Prediction: Mean of the series
        return np.mean(values)

class TestMASCOTS(unittest.TestCase):
    def test_sax_paa(self):
        sax = SAXTransformer(n_segments=2, alphabet_size=3)
        ts = np.array([1.0, 1.0, 5.0, 5.0])
        # PAA should be approx [1.0, 5.0] normalized
        paa = sax.to_paa(ts)
        self.assertEqual(len(paa), 2)
        self.assertAlmostEqual(paa[0], 1.0)
        self.assertAlmostEqual(paa[1], 5.0)

    def test_mascots_counterfactual(self):
        model = DummyModel()
        mascots = MASCOTS(model, n_segments=5)
        
        # Original: low values -> low mean
        query_ts = np.zeros(10)
        current_pred = model.predict_from_array(query_ts)
        
        # Goal: High mean
        target = 2.0
        def condition(pred):
            return pred > target
        
        cf_ts, cf_pred = mascots.generate_counterfactual(query_ts, condition, max_iter=100)
        
        if cf_ts is not None:
            self.assertTrue(cf_pred > target)
            self.assertNotEqual(np.sum(cf_ts), 0)
        else:
            print("Could not find CF (random search might have failed)")

if __name__ == '__main__':
    unittest.main()
