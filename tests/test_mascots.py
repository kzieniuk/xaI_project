import unittest
import numpy as np
from src.mascots import SAXTransformer, MascotsExplainer

class DummyModel:
    def predict_from_array(self, values):
        # Simple Prediction: Mean of the series
        return np.mean(values)

    def predict_batch(self, batch_values):
        # Mean per row
        return np.mean(batch_values, axis=1)

class TestMASCOTS(unittest.TestCase):
    def test_sax_transform_shape(self):
        sax = SAXTransformer(n_segments=2, alphabet_size=3)
        ts = np.array([1.0, 1.0, 5.0, 5.0])
        symbols = sax.transform(ts.reshape(1, -1))
        self.assertEqual(len(symbols), 1)
        self.assertEqual(len(symbols[0]), 2)

    def test_mascots_fit_and_explain_runs(self):
        model = DummyModel()
        explainer = MascotsExplainer(model, n_segments=5, alphabet_size=5, ngram=2)

        rng = np.random.default_rng(0)
        training_windows = rng.normal(size=(64, 20)).astype(np.float32)
        explainer.fit(training_windows, sample_size=64)

        query_ts = np.zeros(20, dtype=np.float32)
        cf_ts, cf_pred = explainer.explain(query_ts, target_class=1)
        # Counterfactual search may fail (algorithm is heuristic), but it must run.
        self.assertTrue(cf_ts is None or cf_ts.shape == query_ts.shape)
        self.assertTrue(cf_pred is None or np.isfinite(cf_pred))

if __name__ == '__main__':
    unittest.main()
