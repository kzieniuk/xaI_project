import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import Counter
from sklearn.linear_model import LogisticRegression

class SAXTransformer:
    def __init__(self, n_segments=10, alphabet_size=5):
        self.n_segments = n_segments
        self.alphabet_size = alphabet_size
        # Breakpoints for equal-probability bins under N(0,1)
        self.breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

    def _segment_edges(self, original_len: int) -> np.ndarray:
        if original_len <= 0:
            raise ValueError("original_len must be > 0")
        if self.n_segments <= 0:
            raise ValueError("n_segments must be > 0")
        # Distribute remainder across segments deterministically
        return np.linspace(0, original_len, self.n_segments + 1, dtype=int)

    def symbol_interval_z(self, symbol: str) -> tuple[float, float]:
        """Return (lower, upper) bounds in z-space for a SAX symbol."""
        if len(symbol) != 1:
            raise ValueError("symbol must be a single character")
        idx = ord(symbol) - 97
        if idx < 0 or idx >= self.alphabet_size:
            raise ValueError(f"symbol out of alphabet range: {symbol}")

        q_lower = idx / self.alphabet_size
        q_upper = (idx + 1) / self.alphabet_size

        lower = float(norm.ppf(q_lower)) if q_lower > 0 else float("-inf")
        upper = float(norm.ppf(q_upper)) if q_upper < 1 else float("inf")
        return lower, upper

    def symbol_centroid_z(self, symbol: str) -> float:
        """Return a representative centroid in z-space for a SAX symbol."""
        if len(symbol) != 1:
            raise ValueError("symbol must be a single character")
        idx = ord(symbol) - 97
        if idx < 0 or idx >= self.alphabet_size:
            raise ValueError(f"symbol out of alphabet range: {symbol}")
        q = (idx + 0.5) / self.alphabet_size
        return float(norm.ppf(q))

    def transform(self, X):
        """
        X: numpy array of shape (n_samples, input_len)
        Returns: list of strings (SAX representations)
        """
        sax_strings = []
        for ts in X:
            # Z-normalize
            if np.std(ts) == 0:
                ts_norm = ts - np.mean(ts)
            else:
                ts_norm = (ts - np.mean(ts)) / (np.std(ts) + 1e-9)
            
            # PAA
            n = len(ts)
            edges = self._segment_edges(n)
            paa = []
            for i in range(self.n_segments):
                start = int(edges[i])
                end = int(edges[i + 1])
                paa.append(np.mean(ts_norm[start:end]))
            
            # SAX
            string = []
            for val in paa:
                idx = np.searchsorted(self.breakpoints, val)
                string.append(chr(97 + idx))
            sax_strings.append("".join(string))
            
        return sax_strings

    def reconstruct(self, sax_string, original_len):
        """
        Approximate reconstruction from SAX string (inverse PAA + simple upsampling)
        """
        paa = [self.symbol_centroid_z(char) for char in sax_string]
        edges = self._segment_edges(original_len)
        out = np.empty(original_len, dtype=np.float32)
        for i, val in enumerate(paa):
            start = int(edges[i])
            end = int(edges[i + 1])
            out[start:end] = val
        return out

    def reconstruct_denorm(self, sax_string: str, original_len: int, mean: float, std: float) -> np.ndarray:
        """Reconstruct a numeric series in original units using provided mean/std."""
        z = self.reconstruct(sax_string, original_len)
        if std == 0:
            return (z + mean).astype(np.float32)
        return (z * std + mean).astype(np.float32)


class MascotsExplainer:
    def __init__(self, blackbox_model, n_segments=10, alphabet_size=5, ngram=3):
        self.blackbox_model = blackbox_model
        self.sax = SAXTransformer(n_segments, alphabet_size)
        self.ngram = ngram
        self.vocab = None
        self.surrogate = None
        self.fitted = False

    def _borf(self, symbols_list):
        """
        Create Bags of Receptive Fields (n-grams)
        symbols_list: list of SAX strings
        Returns: list of Counters (bags)
        """
        bags = []
        for symbols in symbols_list:
            # Generate n-grams
            ngrams = [tuple(symbols[i:i+self.ngram]) for i in range(len(symbols) - self.ngram + 1)]
            bags.append(Counter(ngrams))
        return bags

    def _vectorize(self, bags, vocab_list):
        """
        Convert bags to vector based on vocab
        """
        # Map vocab tuple -> index
        vocab_map = {v: i for i, v in enumerate(vocab_list)}
        n_vocab = len(vocab_list)
        
        vectors = []
        for bag in bags:
            vec = np.zeros(n_vocab)
            for gram, count in bag.items():
                if gram in vocab_map:
                    vec[vocab_map[gram]] = count
            vectors.append(vec)
        return np.vstack(vectors)

    def fit(self, training_windows, sample_size=1000, random_state=None):
        """
        Train the surrogate model.
        training_windows: (N, T) array
        """
        rng = np.random.default_rng(random_state)
        if len(training_windows) > sample_size:
            idx = rng.choice(len(training_windows), size=sample_size, replace=False)
            X_train = training_windows[idx]
        else:
            X_train = training_windows

        print(f"Fitting MASCOTS Surrogate on {len(X_train)} samples...")
        
        # 1. Prediction (Blackbox)
        # We need a binary target! The user snippet uses LogisticRegression.
        # Our model is regression (output float).
        # We must define "classes" implicitly or assume the User provides a split?
        # Let's infer a binary split for the surrogate: High vs Low
        # Or better: The user wants to explain a specific "High" vs "Low" move.
        # But `fit` is general.
        # We will split by Median for general training, or perhaps just threshold 0.
        
        preds = []
        # Batch predict for speed
        # Simple manual batch
        batch_size = 256
        for i in range(0, len(X_train), batch_size):
            chunk = X_train[i:i+batch_size]
            p = self.blackbox_model.predict_batch(chunk)
            preds.append(p)
        preds = np.concatenate(preds)
        
        # Binarize: > 0 (Positive Return) vs <= 0 (Negative Return)
        y_surrogate = (preds > 0).astype(int)
        
        # 2. SAX + BoRF
        sax_strings = self.sax.transform(X_train)
        bags = self._borf(sax_strings)
        
        # Build Vocab
        all_grams = set().union(*[b.keys() for b in bags])
        self.vocab = sorted(list(all_grams))
        self._vocab_map = {v: i for i, v in enumerate(self.vocab)}
        
        X_vec = self._vectorize(bags, self.vocab)
        
        # 3. Train Surrogate
        self.surrogate = LogisticRegression(max_iter=1000, C=1.0)
        self.surrogate.fit(X_vec, y_surrogate)
        
        acc = self.surrogate.score(X_vec, y_surrogate)
        print(f"Surrogate Fit Complete. Accuracy vs Blackbox Sign: {acc:.2%}")
        self.fitted = True

    def explain(
        self,
        query_ts,
        target_class=None,
        max_harmful_grams=20,
        tries_per_gram=10,
        random_state=None,
        return_details: bool = False,
    ):
        """
        Generate counterfactual for query_ts to flipped class.
        """
        if not self.fitted:
            raise ValueError("Must call fit() before explain()")
            
        # Current Prediction
        orig_pred_val = self.blackbox_model.predict_from_array(query_ts)
        orig_class = 1 if orig_pred_val > 0 else 0
        
        if target_class is None:
            target_class = 1 - orig_class
            
        print(f"Explaining: Orig ({orig_pred_val:.4f}, Class {orig_class}) -> Target Class {target_class}")
        
        query_ts = np.asarray(query_ts, dtype=np.float32)
        # Symbolize Query
        query_sax = self.sax.transform(query_ts.reshape(1, -1))[0]
        query_bag = self._borf([query_sax])[0]
        # query_vec = self._vectorize([query_bag], self.vocab)
        
        # Identify Important Features (n-grams) for current class
        # Ideally, we look at weights that push towards the OPPOSITE class?
        # Or weights that push towards CURRENT class and remove them?
        # User snippet: "weights = surrogate.coef_[0]", "important = np.argsort(np.abs(weights))"
        # It iterates important features and RANDOMIZES them.
        
        weights = self.surrogate.coef_[0] # Shape (n_vocab,)
        
        # If target is 1 (Positive), we want positive weights.
        # If target is 0 (Negative), we want negative weights.
        # Wait, user snippet strategy:
        # "Replace important segments with random class symbols"
        
        # Improved Strategy:
        # Find n-grams in the query that contribute most to the WRONG class (Original Class)
        # And replace them.
        
        rng = None
        if random_state is not None:
            import random as _random
            rng = _random.Random(random_state)

        # Get indices of grams present in query
        present_indices = []
        for gram, count in query_bag.items():
            if gram in self._vocab_map:
                idx = self._vocab_map[gram]
                present_indices.append(idx)
                
        # Filter weights by presence
        # We want to identify grams driving the prediction away from target
        # If target is 1, we want to remove grams with highly NEGATIVE weight.
        # If target is 0, we want to remove grams with highly POSITIVE weight.
        
        relevant_indices = []
        for idx in present_indices:
            w = weights[idx]
            if target_class == 1 and w < 0: # Driving down
                relevant_indices.append((idx, w))
            elif target_class == 0 and w > 0: # Driving up
                relevant_indices.append((idx, w))
                
        # Sort by magnitude (most harmful first)
        relevant_indices.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Try swapping
        cf_sax_list = list(query_sax)

        # For denormalization of z-centroids back into the query_ts scale
        ts_mean = float(np.mean(query_ts))
        ts_std = float(np.std(query_ts))

        edges = self.sax._segment_edges(len(query_ts))
        
        # Patterns to inject? 
        # User snippet: "np.random.randint" (random symbols).
        # We can try replacing the "harmful" n-gram with a "neutral" or "helpful" one?
        # Simpler: Just randomize the pattern for now, or replace with 'aaaaa' (flat?)
        
        # Let's try replacing with a random "valid" pattern from vocab that has good weight?
        # Find "good" patterns
        if target_class == 1:
            good_gram_indices = np.where(weights > 0)[0]
        else:
            good_gram_indices = np.where(weights < 0)[0]
            
        good_grams = [self.vocab[i] for i in good_gram_indices]
        if not good_grams:
            good_grams = self.vocab # Fallback
            
        best_cf_ts = None
        
        # Limit attempts
        for idx, w in relevant_indices[:max_harmful_grams]:
            bad_gram = self.vocab[idx] # tuple of chars
            # Find where this gram occurs in query
            # A gram is length ngram (3)
            # Search in char list
            
            # Simple substring search
            gram_len = len(bad_gram)
            
            # There might be multiple occurrences, let's swap the first one found for now
            # Convert list back to string for find
            curr_str = "".join(cf_sax_list)
            bad_gram_str = "".join(bad_gram)
            pos = curr_str.find(bad_gram_str)
            
            if pos != -1:
                # Swap!
                # Pick a random "good" gram
                for _ in range(max(1, tries_per_gram)):
                    if rng is None:
                        import random
                        replacement_gram = random.choice(good_grams)
                    else:
                        replacement_gram = rng.choice(good_grams)
                
                    # Apply swap in SAX space
                    for k in range(gram_len):
                        cf_sax_list[pos + k] = replacement_gram[k]

                    # Reconstruct continuous TS by modifying only affected segments
                    cf_ts = query_ts.copy()

                    changed_segments = []
                    for k in range(gram_len):
                        sax_idx = pos + k
                        old_char = query_sax[sax_idx]
                        new_char = replacement_gram[k]

                        start = int(edges[sax_idx])
                        end = int(edges[sax_idx + 1])

                        z_val = self.sax.symbol_centroid_z(new_char)
                        if ts_std == 0:
                            val_denorm = float(z_val + ts_mean)
                        else:
                            val_denorm = float(z_val * ts_std + ts_mean)
                        cf_ts[start:end] = val_denorm

                        if return_details:
                            old_lower, old_upper = self.sax.symbol_interval_z(old_char)
                            new_lower, new_upper = self.sax.symbol_interval_z(new_char)
                            changed_segments.append(
                                {
                                    "segment_index": int(sax_idx),
                                    "start": int(start),
                                    "end": int(end),
                                    "from_symbol": old_char,
                                    "to_symbol": new_char,
                                    "from_interval_z": (float(old_lower), float(old_upper)),
                                    "to_interval_z": (float(new_lower), float(new_upper)),
                                    "original_segment_mean": float(np.mean(query_ts[start:end])),
                                    "counterfactual_value": float(val_denorm),
                                }
                            )

                    new_pred = self.blackbox_model.predict_from_array(cf_ts)
                    new_class = 1 if new_pred > 0 else 0

                    if new_class == target_class:
                        print(
                            f"  Counterfactual Found! Swapped '{bad_gram_str}' with '{''.join(replacement_gram)}'"
                        )
                        print(f"  New Pred: {new_pred:.4f}")
                        if not return_details:
                            return cf_ts, new_pred

                        details = {
                            "query_sax": query_sax,
                            "counterfactual_sax": "".join(cf_sax_list),
                            "swap": {
                                "pos": int(pos),
                                "bad_gram": bad_gram_str,
                                "replacement_gram": "".join(replacement_gram),
                            },
                            "changed_segments": changed_segments,
                            "sax": {
                                "n_segments": int(self.sax.n_segments),
                                "alphabet_size": int(self.sax.alphabet_size),
                                "ngram": int(self.ngram),
                            },
                        }
                        return cf_ts, new_pred, details

                    # Reset for next try
                    cf_sax_list = list(query_sax)
                    
        print("No counterfactual found.")
        if return_details:
            return None, None, None
        return None, None
