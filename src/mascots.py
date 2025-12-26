import numpy as np
import pandas as pd
from scipy.stats import norm

class SAXTransformer:
    def __init__(self, n_segments=5, alphabet_size=5):
        self.n_segments = n_segments
        self.alphabet_size = alphabet_size
        self.breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

    def to_paa(self, ts):
        n = len(ts)
        segment_len = n // self.n_segments
        paa = []
        for i in range(self.n_segments):
            start = i * segment_len
            end = start + segment_len
            paa.append(np.mean(ts[start:end]))
        return np.array(paa)

    def to_sax(self, ts):
        # Z-normalize
        ts_norm = (ts - np.mean(ts)) / (np.std(ts) + 1e-6)
        paa = self.to_paa(ts_norm)
        sax_string = []
        for val in paa:
            # Find bucket
            idx = np.searchsorted(self.breakpoints, val)
            sax_string.append(chr(97 + idx)) # 'a', 'b', ...
        return "".join(sax_string), paa

    def from_sax_paa(self, paa, original_len):
        # Reconstruct approximate TS from PAA
        # Simple upsampling
        segment_len = original_len // self.n_segments
        ts_recon = np.repeat(paa, segment_len)
        return ts_recon

class MASCOTS:
    def __init__(self, model, n_segments=10, alphabet_size=10):
        self.model = model
        self.sax = SAXTransformer(n_segments, alphabet_size)

    def generate_counterfactual(self, query_ts, desired_outcome_fn, max_iter=1000):
        """
        query_ts: original time series (numpy array)
        desired_outcome_fn: function(prediction) -> bool
        Returns: counterfactual_ts, new_pred
        """
        original_string, original_paa = self.sax.to_sax(query_ts)
        current_paa = original_paa.copy()
        current_ts = query_ts.copy()
        
        best_cf = None
        best_dist = float('inf')
        
        print(f"Searching for counterfactual... Original Pred: {self.model.predict_from_array(query_ts)}")

        for i in range(max_iter):
            # Perturb PAA
            # Select random segment
            idx = np.random.randint(0, len(current_paa))
            # Perturb value
            perturbation = np.random.normal(0, 0.5)
            candidate_paa = current_paa.copy()
            candidate_paa[idx] += perturbation
            
            # Reconstruct (inverse PAA + denormalize logic if needed, but we work in normalized space for model usually)
            # Simplification: Model assumes normalized data or handles it. 
            # We reconstruct shape 
            candidate_ts_shape = self.sax.from_sax_paa(candidate_paa, len(query_ts))
            
            # Apply shape trend to original mean/std or just use shape if model trained on normalized
            # Assuming model handles raw, we might want to scale back.
            # For this MVP, let's assume we modify the raw values roughly by adding the localized change
            
            # Better: Modify the original TS directly at the segment corresponding to the PAA change
            segment_len = len(query_ts) // self.sax.n_segments
            start = idx * segment_len
            end = start + segment_len
            
            candidate_ts = current_ts.copy()
            candidate_ts[start:end] += perturbation # Add perturbation to the raw segment
            
            pred = self.model.predict_from_array(candidate_ts)
            
            if desired_outcome_fn(pred):
                dist = np.linalg.norm(candidate_ts - query_ts)
                if dist < best_dist:
                    best_dist = dist
                    best_cf = candidate_ts
                    current_ts = candidate_ts # Greedy step: move towards valid regions
                    current_paa = candidate_paa
                    # We can stop early or continue optimization
                    # Let's simple return first valid found for speed in detailed task
                    return best_cf, pred
            
        return best_cf, None # Failed
