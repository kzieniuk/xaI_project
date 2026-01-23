import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import Counter
from sklearn.linear_model import LogisticRegression

class SAXTransformer:
    def __init__(self, n_segments=10, alphabet_size=5):
        self.n_segments = n_segments
        self.alphabet_size = alphabet_size
        self.breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

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
            segment_len = n // self.n_segments
            paa = []
            for i in range(self.n_segments):
                start = i * segment_len
                end = start + segment_len
                # Handle leftovers in last segment if needed, but assuming simple division
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
        paa = []
        for char in sax_string:
            idx = ord(char) - 97
            # Map index back to centroid of interval (approx)
            # Simple assumption: uniform mapping to normal distribution centers?
            # Or simplified: map to breakpoint midpoints
            lower = self.breakpoints[idx-1] if idx > 0 else -2.0
            upper = self.breakpoints[idx] if idx < len(self.breakpoints) else 2.0
            val = (lower + upper) / 2.0
            paa.append(val)
            
        segment_len = original_len // self.n_segments
        return np.repeat(paa, segment_len)


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

    def fit(self, training_windows, sample_size=1000):
        """
        Train the surrogate model.
        training_windows: (N, T) array
        """
        if len(training_windows) > sample_size:
            idx = np.random.choice(len(training_windows), sample_size, replace=False)
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
        
        X_vec = self._vectorize(bags, self.vocab)
        
        # 3. Train Surrogate
        self.surrogate = LogisticRegression(max_iter=1000, C=1.0)
        self.surrogate.fit(X_vec, y_surrogate)
        
        acc = self.surrogate.score(X_vec, y_surrogate)
        print(f"Surrogate Fit Complete. Accuracy vs Blackbox Sign: {acc:.2%}")
        self.fitted = True

    def explain(self, query_ts, target_class=None, max_harmful_grams=50, tries_per_gram=20, random_state=None, return_details=False):
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
        
        # Get indices of grams present in query
        present_indices = []
        for gram, count in query_bag.items():
            if gram in self.vocab:
                idx = self.vocab.index(gram)
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
            
        valid_cfs = []
        
        # Limit attempts
        for idx, w in relevant_indices[:max_harmful_grams]: # Try top N harmful grams
            bad_gram = self.vocab[idx] # tuple of chars
            
            # Helper for centroid logic
            def get_centroid_vec(gram_tuple):
                vec = []
                for char in gram_tuple:
                    idx = ord(char) - 97
                    lower = self.sax.breakpoints[idx-1] if idx > 0 else -2.0
                    upper = self.sax.breakpoints[idx] if idx < len(self.sax.breakpoints) else 2.0
                    vec.append((lower + upper) / 2.0)
                return np.array(vec)

            # Control Perturbation: Find CLOSEST good gram
            bad_vec = get_centroid_vec(bad_gram)
            
            # Sort good_grams by distance to bad_gram
            good_grams_sorted = sorted(good_grams, key=lambda g: np.linalg.norm(bad_vec - get_centroid_vec(g)))
            
            # Pick from top K closest to allow some exploration (tries_per_gram)
            candidates = good_grams_sorted[:tries_per_gram]

            # Find where this gram occurs in query
            gram_len = len(bad_gram)
            
            curr_str = "".join(cf_sax_list)
            bad_gram_str = "".join(bad_gram)
            pos = curr_str.find(bad_gram_str)
            
            if pos != -1:
                # Batch Prediction Optimization
                candidate_ts_list = []
                candidate_grams = []
                
                for replacement_gram in candidates:
                    # Apply swap in SAX space
                    cf_sax_list_try = list(cf_sax_list) 
                    for k in range(gram_len):
                        cf_sax_list_try[pos+k] = replacement_gram[k]
                    
                    # Reconstruct continuous TS
                    cf_ts = query_ts.copy()
                    segment_len = len(query_ts) // self.sax.n_segments
                    
                    for k in range(gram_len):
                        sax_idx = pos + k
                        char = replacement_gram[k]
                        
                        start = sax_idx * segment_len
                        end = start + segment_len
                        
                        char_idx = ord(char) - 97
                        lower = self.sax.breakpoints[char_idx-1] if char_idx > 0 else -2.0
                        upper = self.sax.breakpoints[char_idx] if char_idx < len(self.sax.breakpoints) else 2.0
                        val = (lower + upper) / 2.0
                        
                        if np.std(query_ts) == 0:
                             val_denorm = val + np.mean(query_ts)
                        else:
                             val_denorm = (val * np.std(query_ts)) + np.mean(query_ts)
                        
                        cf_ts[start:end] = val_denorm
                    
                    candidate_ts_list.append(cf_ts)
                    candidate_grams.append(replacement_gram)
                
                # Predict Batch
                if candidate_ts_list:
                    batch_preds = self.blackbox_model.predict_batch(np.array(candidate_ts_list))
                    
                    for i, new_pred in enumerate(batch_preds):
                        new_class = 1 if new_pred > 0 else 0
                        
                        if new_class == target_class:
                           replacement_gram = candidate_grams[i]
                           print(f"  Candidate CF found! Swapped '{bad_gram_str}' with '{''.join(replacement_gram)}'")
                           valid_cfs.append((candidate_ts_list[i], new_pred))
            
            # Optimization: If we have enough candidates, stop searching to save time?
            if len(valid_cfs) >= 10:
                print("  Collected 10 valid candidates. Selecting best...")
                break
        
        if valid_cfs:
            print(f"Selecting best from {len(valid_cfs)} candidates based on Weighted MSE...")
            # Selection Metric: Weighted MSE
            # Weight increases closer to T (end of array)
            T = len(query_ts)
            weights = np.linspace(0.1, 10.0, T) # Strongly penalize recent changes
            # Normalize? Not strictly needed for comparison
            
            best_cf = None
            best_score = float('inf')
            best_pred_val = 0
            
            for cf, pred_val in valid_cfs:
                # Weighted MSE
                sq_diff = (cf - query_ts) ** 2
                weighted_mse = np.mean(sq_diff * weights)
                
                if weighted_mse < best_score:
                    best_score = weighted_mse
                    best_cf = cf
                    best_pred_val = pred_val
            
            print(f"  Selected best CF with Weighted Score: {best_score:.4f}")
            return best_cf, best_pred_val
            
        print("No counterfactual found.")
        return None, None
