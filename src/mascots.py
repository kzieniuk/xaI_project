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

    def explain(self, query_ts, target_class=None):
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
            
        best_cf_ts = None
        
        # Limit attempts
        for idx, w in relevant_indices[:5]: # Try top 5 harmful grams
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
                import random
                replacement_gram = random.choice(good_grams)
                
                # Apply swap in SAX space
                # Update list
                for k in range(gram_len):
                    cf_sax_list[pos+k] = replacement_gram[k]
                
                # Reconstruct and Test
                new_sax_str = "".join(cf_sax_list)
                
                # Reconstruct continuous TS
                # We need to map the NEW sax string back to values.
                # But simple reconstruction destroys the original info of untouched segments!
                # Better: Modify ONLY the swapped segments in the original TS.
                
                cf_ts = query_ts.copy()
                
                # Determine indices in TS corresponding to SAX segment
                # SAX segment i covers indices [i*seg_len : (i+1)*seg_len]
                segment_len = len(query_ts) // self.sax.n_segments
                
                for k in range(gram_len):
                    sax_idx = pos + k
                    char = replacement_gram[k]
                    
                    # Reconstruct value for this specific segment
                    start = sax_idx * segment_len
                    end = start + segment_len
                    
                    # Get value from reconstructing just this char
                    # Use breakpoint midpoint
                    char_idx = ord(char) - 97
                    lower = self.sax.breakpoints[char_idx-1] if char_idx > 0 else -2.0
                    upper = self.sax.breakpoints[char_idx] if char_idx < len(self.sax.breakpoints) else 2.0
                    val = (lower + upper) / 2.0
                    
                    # De-normalize? We assumed TS was Z-normed for SAX.
                    # We need mean/std of original query_ts to reverse.
                    if np.std(query_ts) == 0:
                         val_denorm = val + np.mean(query_ts)
                    else:
                         val_denorm = (val * np.std(query_ts)) + np.mean(query_ts)
                    
                    # Check bounds
                    # Flat fill
                    cf_ts[start:end] = val_denorm
                
                # Check prediction
                new_pred = self.blackbox_model.predict_from_array(cf_ts)
                new_class = 1 if new_pred > 0 else 0
                
                if new_class == target_class:
                    print(f"  Counterfactual Found! Swapped '{bad_gram_str}' with '{''.join(replacement_gram)}'")
                    print(f"  New Pred: {new_pred:.4f}")
                    return cf_ts, new_pred
                else:
                    # Revert for next try? Or keep evolving?
                    # Greedy: revert if didn't help?
                    # For simplicity, revert to try next harmful gram independently
                    cf_sax_list = list(query_sax) # Reset
                    
        print("No counterfactual found.")
        return None, None
