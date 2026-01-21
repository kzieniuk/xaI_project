import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable

from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Pattern:
    channel: int
    word: str  # SAX word, length = n_paa


class SAXWords:
    """SAX word extraction for fixed window length.

    This is a minimal, paper-aligned building block:
    - z-normalize a subsequence (local mean/std)
    - PAA into n_paa segments
    - map segment means to SAX symbols via Gaussian breakpoints

    Note: In the paper, PatternSwap uses local subsequence mean/std.
    We follow that for PatternSwap; for word extraction we also use local z-norm.
    """

    def __init__(self, n_paa: int = 8, alphabet_size: int = 5):
        if n_paa <= 0:
            raise ValueError("n_paa must be > 0")
        if alphabet_size <= 1:
            raise ValueError("alphabet_size must be > 1")
        self.n_paa = int(n_paa)
        self.alphabet_size = int(alphabet_size)
        self.breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])

    def symbol_centroid_z(self, symbol: str) -> float:
        idx = ord(symbol) - 97
        if idx < 0 or idx >= self.alphabet_size:
            raise ValueError(f"symbol out of alphabet range: {symbol}")
        q = (idx + 0.5) / self.alphabet_size
        return float(norm.ppf(q))

    def _paa_edges(self, w: int) -> np.ndarray:
        # deterministic equal-width segmentation (may include unequal sizes by 1)
        return np.linspace(0, w, self.n_paa + 1, dtype=int)

    def word_for_subseq(self, subseq: np.ndarray) -> str:
        subseq = np.asarray(subseq, dtype=np.float32)
        w = int(subseq.shape[0])
        mu = float(np.mean(subseq))
        sigma = float(np.std(subseq))
        if sigma == 0:
            z = subseq - mu
        else:
            z = (subseq - mu) / (sigma + 1e-9)

        edges = self._paa_edges(w)
        paa = []
        for i in range(self.n_paa):
            s = int(edges[i])
            e = int(edges[i + 1])
            paa.append(float(np.mean(z[s:e])))

        chars = []
        for v in paa:
            idx = int(np.searchsorted(self.breakpoints, v))
            chars.append(chr(97 + idx))
        return "".join(chars)

    def paa_means_z(self, subseq: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Return (paa_means_z, mu, sigma) for a subsequence."""
        subseq = np.asarray(subseq, dtype=np.float32)
        w = int(subseq.shape[0])
        mu = float(np.mean(subseq))
        sigma = float(np.std(subseq))
        if sigma == 0:
            z = subseq - mu
        else:
            z = (subseq - mu) / (sigma + 1e-9)

        edges = self._paa_edges(w)
        paa = np.empty(self.n_paa, dtype=np.float32)
        for i in range(self.n_paa):
            s = int(edges[i])
            e = int(edges[i + 1])
            paa[i] = float(np.mean(z[s:e]))
        return paa, mu, sigma


class BoRF:
    """A minimal BoRF-like vectorizer for multivariate sequences.

    For each channel, extracts SAX words from sliding windows of length w.
    The feature space is (channel, word). The z-vector is counts of words.

    Additionally returns an occurrence index to support align(channel, word).

    This is not a full external BoRF implementation, but matches the needs of
    Algorithm 1/2 in the paper: hash/inverse-hash, z_k != 0 constraint, and align.
    """

    def __init__(self, sax: SAXWords, w: int = 24, stride: int = 1):
        if w <= 0:
            raise ValueError("w must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        self.sax = sax
        self.w = int(w)
        self.stride = int(stride)

        self.vocab: list[Pattern] = []
        self._index: dict[Pattern, int] = {}

    def fit(self, X: np.ndarray) -> None:
        """Build vocab from training set.

        X: (n, d, m)
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("X must be (n, d, m)")
        n, d, m = X.shape
        if m < self.w:
            raise ValueError("m must be >= w")

        patterns: set[Pattern] = set()
        for i in range(n):
            for ch in range(d):
                ts = X[i, ch]
                for t in range(0, m - self.w + 1, self.stride):
                    word = self.sax.word_for_subseq(ts[t : t + self.w])
                    patterns.add(Pattern(channel=ch, word=word))

        self.vocab = sorted(patterns, key=lambda p: (p.channel, p.word))
        self._index = {p: i for i, p in enumerate(self.vocab)}

    def transform_one(self, x: np.ndarray) -> tuple[np.ndarray, dict[int, list[int]]]:
        """Return (z_counts, occurrences).

        x: (d, m)
        z_counts: (r,)
        occurrences: feature_index -> list of start indices where the pattern occurs
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("x must be (d, m)")
        d, m = x.shape
        if m < self.w:
            raise ValueError("m must be >= w")
        if not self._index:
            raise ValueError("BoRF must be fit() before transform")

        z = np.zeros(len(self.vocab), dtype=np.int32)
        occ: dict[int, list[int]] = {}
        for ch in range(d):
            ts = x[ch]
            for t in range(0, m - self.w + 1, self.stride):
                word = self.sax.word_for_subseq(ts[t : t + self.w])
                p = Pattern(channel=ch, word=word)
                k = self._index.get(p)
                if k is None:
                    continue
                z[k] += 1
                occ.setdefault(k, []).append(int(t))
        return z, occ

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("X must be (n, d, m)")
        out = np.zeros((X.shape[0], len(self.vocab)), dtype=np.int32)
        for i in range(X.shape[0]):
            out[i], _ = self.transform_one(X[i])
        return out

    def inverse_hash(self, k: int) -> Pattern:
        return self.vocab[int(k)]


@dataclass
class MascotsPaperDetails:
    predicted_class: int
    iterations: int
    applied: list[dict]


class MascotsPaperExplainer:
    """Paper-aligned MASCOTS core loop (Algorithm 1/2) for a binary blackbox.

    - b(X): blackbox that returns a scalar score; class is (score>0)
    - g: LogisticRegression surrogate on BoRF counts
    - e: for LR we use per-instance relevance = coef * z (class-specific)
    - GetPerturbation: choose k+ among present patterns; choose k- by relevance+lambda*L1
    - align: pick random occurrence in the time series
    - PatternSwap: enforce SAX word swap on that local subsequence via piecewise-constant segment targets
    - iterative update until class flips or max_iters reached
    """

    def __init__(
        self,
        blackbox_predict_one: Callable[[np.ndarray], float],
        blackbox_predict_batch: Callable[[np.ndarray], np.ndarray],
        w: int = 24,
        n_paa: int = 8,
        alphabet_size: int = 5,
        stride: int = 1,
        lam: float = 0.5,
        max_iters: int = 12,
        random_state: int | None = None,
    ):
        self.blackbox_predict_one = blackbox_predict_one
        self.blackbox_predict_batch = blackbox_predict_batch
        self.sax = SAXWords(n_paa=n_paa, alphabet_size=alphabet_size)
        self.borf = BoRF(self.sax, w=w, stride=stride)
        self.lam = float(lam)
        self.max_iters = int(max_iters)
        self.rng = np.random.default_rng(random_state)

        self.surrogate = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(self, X_train: np.ndarray, sample_size: int = 512) -> None:
        """Fit BoRF vocab + surrogate.

        X_train: (n, d, m) in the *modifiable feature space* (selected channels only)
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        if X_train.ndim != 3:
            raise ValueError("X_train must be (n, d, m)")

        if X_train.shape[0] > sample_size:
            idx = self.rng.choice(X_train.shape[0], size=sample_size, replace=False)
            Xs = X_train[idx]
        else:
            Xs = X_train

        # blackbox labels
        scores = self.blackbox_predict_batch(Xs).astype(np.float32)
        y = (scores > 0).astype(int)

        self.borf.fit(Xs)
        Z = self.borf.transform(Xs).astype(np.float32)
        self.surrogate.fit(Z, y)
        self._fitted = True

    def _relevance(self, z: np.ndarray, predicted_class: int) -> np.ndarray:
        # LR coef_: (1, r) for binary; treat it as class-1 logit weight.
        w = self.surrogate.coef_[0].astype(np.float32)
        # relevance toward class 1 is w*z; toward class 0 is (-w)*z
        if predicted_class == 1:
            return w * z
        return (-w) * z

    def _pattern_vec(self, word: str) -> np.ndarray:
        # Map SAX symbols to their z-centroids
        return np.asarray([self.sax.symbol_centroid_z(c) for c in word], dtype=np.float32)

    def _pattern_swap_inplace(self, x: np.ndarray, ch: int, t0: int, p_to: str) -> dict:
        """Apply PatternSwap on channel ch, starting at t0, for window length w.

        Returns details in ambient space of the modified subsequence.
        """
        w = self.borf.w
        n_paa = self.sax.n_paa
        seg_edges = np.linspace(0, w, n_paa + 1, dtype=int)

        subseq = x[ch, t0 : t0 + w].astype(np.float32)
        paa_z, mu, sigma = self.sax.paa_means_z(subseq)

        targets_z = np.asarray([self.sax.symbol_centroid_z(c) for c in p_to], dtype=np.float32)
        # Target value per segment in original units: target_z * sigma + mu
        # If sigma==0, keep constant around mu.
        target_vals = targets_z * (sigma if sigma != 0 else 1.0) + mu

        before = subseq.copy()
        for i in range(n_paa):
            s = int(seg_edges[i])
            e = int(seg_edges[i + 1])
            x[ch, t0 + s : t0 + e] = float(target_vals[i])

        after = x[ch, t0 : t0 + w].astype(np.float32)
        return {
            "channel": int(ch),
            "t_start": int(t0),
            "t_end": int(t0 + w),
            "w": int(w),
            "n_paa": int(n_paa),
            "mu": float(mu),
            "sigma": float(sigma),
            "before": before.tolist(),
            "after": after.tolist(),
            "delta": (after - before).tolist(),
        }

    def explain(self, x: np.ndarray, target_class: int | None = None) -> tuple[np.ndarray | None, float | None, MascotsPaperDetails | None]:
        if not self._fitted:
            raise ValueError("Call fit() first")

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("x must be (d, m)")

        # original
        orig_score = float(self.blackbox_predict_one(x))
        orig_class = 1 if orig_score > 0 else 0
        if target_class is None:
            target_class = 1 - orig_class

        x_cf = x.copy()
        applied: list[dict] = []

        for it in range(self.max_iters):
            score = float(self.blackbox_predict_one(x_cf))
            pred_class = 1 if score > 0 else 0
            if pred_class == target_class:
                return x_cf, score, MascotsPaperDetails(predicted_class=pred_class, iterations=it, applied=applied)

            z, occ = self.borf.transform_one(x_cf)
            rel = self._relevance(z.astype(np.float32), predicted_class=pred_class)

            present = np.where(z > 0)[0]
            if present.size == 0:
                break

            # k+ : most relevant present pattern for current prediction
            k_plus = int(present[np.argmax(rel[present])])
            p_plus = self.borf.inverse_hash(k_plus)

            # k- : pattern against prediction with lambda similarity penalty
            p_plus_vec = self._pattern_vec(p_plus.word)
            best_k_minus = None
            best_obj = None
            for k in range(len(self.borf.vocab)):
                p = self.borf.inverse_hash(k)
                # must have same word length by construction
                p_vec = self._pattern_vec(p.word)
                obj = float(rel[k]) + self.lam * float(np.sum(np.abs(p_vec - p_plus_vec)))
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_k_minus = int(k)
            if best_k_minus is None:
                break
            p_minus = self.borf.inverse_hash(best_k_minus)

            # align: choose random occurrence of p_plus
            starts = occ.get(k_plus, [])
            if not starts:
                break
            t0 = int(self.rng.choice(starts))

            change = self._pattern_swap_inplace(x_cf, ch=p_plus.channel, t0=t0, p_to=p_minus.word)
            change.update(
                {
                    "iter": int(it),
                    "k_plus": int(k_plus),
                    "k_minus": int(best_k_minus),
                    "p_plus": {"channel": int(p_plus.channel), "word": p_plus.word},
                    "p_minus": {"channel": int(p_minus.channel), "word": p_minus.word},
                }
            )
            applied.append(change)

        return None, None, None
