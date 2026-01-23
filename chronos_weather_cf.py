import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.mascots import MascotsExplainer


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def load_weather_matrix(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load iTransformer-style weather.csv.

    Returns:
      data_raw: (T, N) float32
      columns: list[str]
      dates: (T,) datetime64
    """
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"]).to_numpy()
    columns = [c for c in df.columns if c != "date"]
    data_raw = df[columns].to_numpy(dtype=np.float32)
    return data_raw, columns, dates


def split_slices(n: int, seq_len: int) -> tuple[slice, slice, slice]:
    """Match upstream Dataset_Custom split boundaries."""
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_val = n - num_train - num_test

    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_val, n]

    train_slice = slice(border1s[0], border2s[0])
    val_slice = slice(border1s[1], border2s[1])
    test_slice = slice(border1s[2], border2s[2])
    return train_slice, val_slice, test_slice


def _select_pipeline(model_id: str, device_map: str):
    from chronos import Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline

    mid = model_id.lower()
    if "chronos-bolt" in mid:
        return ChronosBoltPipeline.from_pretrained(model_id, device_map=device_map)
    if "chronos-2" in mid:
        return Chronos2Pipeline.from_pretrained(model_id, device_map=device_map)
    return ChronosPipeline.from_pretrained(model_id, device_map=device_map)


@dataclass
class ChronosPredictor:
    pipeline: object
    prediction_length: int
    horizon_step: int

    @torch.no_grad()
    def predict_next(self, series_units: np.ndarray) -> float:
        series_units = np.asarray(series_units, dtype=np.float32)
        if series_units.ndim != 1:
            raise ValueError("series_units must be 1D")
        if self.horizon_step < 0 or self.horizon_step >= self.prediction_length:
            raise ValueError("horizon_step out of range")

        inp = torch.tensor(series_units, dtype=torch.float32)
        out = self.pipeline.predict(inp, prediction_length=self.prediction_length)

        # out layouts depend on forecast_type:
        # - quantiles: (B, Q, H)
        # - samples/trajectories: often (B, S, H)
        # - deterministic: (B, H)
        out = out.detach().cpu()
        if out.ndim == 2:
            return float(out[0, self.horizon_step].item())
        if out.ndim == 3:
            # pick median-ish along dim=1 (quantiles or samples)
            mid = out.shape[1] // 2
            return float(out[0, mid, self.horizon_step].item())
        raise ValueError(f"Unexpected forecast tensor shape: {tuple(out.shape)}")

    @torch.no_grad()
    def predict_batch_next(self, batch_series_units: np.ndarray) -> np.ndarray:
        batch_series_units = np.asarray(batch_series_units, dtype=np.float32)
        if batch_series_units.ndim != 2:
            raise ValueError("batch_series_units must be (B, T)")
        if self.horizon_step < 0 or self.horizon_step >= self.prediction_length:
            raise ValueError("horizon_step out of range")

        inp = torch.tensor(batch_series_units, dtype=torch.float32)
        out = self.pipeline.predict(inp, prediction_length=self.prediction_length)
        out = out.detach().cpu()
        if out.ndim == 2:
            return out[:, self.horizon_step].numpy().astype(np.float32)
        if out.ndim == 3:
            mid = out.shape[1] // 2
            return out[:, mid, self.horizon_step].numpy().astype(np.float32)
        raise ValueError(f"Unexpected forecast tensor shape: {tuple(out.shape)}")


class JumpScoreWrapper:
    """Convert a predictor that outputs next-step value (units) into a scalar event score."""

    def __init__(self, predictor: ChronosPredictor, threshold: float, mode: str = "abs"):
        self.predictor = predictor
        self.threshold = float(threshold)
        self.mode = str(mode)

    def _score(self, pred: float, last_obs: float) -> float:
        delta = float(pred) - float(last_obs)
        if self.mode == "up":
            return delta - self.threshold
        if self.mode == "down":
            return (-delta) - self.threshold
        if self.mode == "abs":
            return abs(delta) - self.threshold
        raise ValueError(f"Unknown jump mode: {self.mode}")

    def predict_from_array(self, history_units: np.ndarray) -> float:
        history_units = np.asarray(history_units, dtype=np.float32)
        last_obs = float(history_units[-1])
        pred = float(self.predictor.predict_next(history_units))
        return float(self._score(pred, last_obs))

    def predict_batch(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be (B, T)")

        out = np.empty((X.shape[0],), dtype=np.float32)
        thr = np.float32(self.threshold)
        for i in range(0, X.shape[0], batch_size):
            chunk = X[i : i + batch_size]
            last = chunk[:, -1].astype(np.float32)
            preds = self.predictor.predict_batch_next(chunk).astype(np.float32)
            delta = preds - last
            if self.mode == "up":
                out[i : i + batch_size] = delta - thr
            elif self.mode == "down":
                out[i : i + batch_size] = (-delta) - thr
            elif self.mode == "abs":
                out[i : i + batch_size] = np.abs(delta) - thr
            else:
                raise ValueError(f"Unknown jump mode: {self.mode}")
        return out


class PredictionShiftScoreWrapper:
    """Score based on changing the model forecast relative to the base (query) forecast.

    Event is defined as abs(pred(x) - base_pred) > threshold.
    """

    def __init__(self, predictor: ChronosPredictor, base_pred: float, threshold: float):
        self.predictor = predictor
        self.base_pred = float(base_pred)
        self.threshold = float(threshold)

    def predict_from_array(self, history_units: np.ndarray) -> float:
        history_units = np.asarray(history_units, dtype=np.float32)
        pred = float(self.predictor.predict_next(history_units))
        return float(abs(pred - self.base_pred) - self.threshold)

    def predict_batch(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be (B, T)")

        out = np.empty((X.shape[0],), dtype=np.float32)
        base = np.float32(self.base_pred)
        thr = np.float32(self.threshold)
        for i in range(0, X.shape[0], batch_size):
            chunk = X[i : i + batch_size]
            preds = self.predictor.predict_batch_next(chunk).astype(np.float32)
            out[i : i + batch_size] = np.abs(preds - base) - thr
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Chronos Weather CF search with MASCOTS (univariate)")
    ap.add_argument("--data", type=str, default="iTransformer_datasets/weather/weather.csv")
    ap.add_argument("--target-col", type=str, default="T (degC)")

    ap.add_argument("--model", type=str, default="amazon/chronos-bolt-tiny")
    ap.add_argument("--device-map", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument(
        "--prediction-length",
        type=int,
        default=0,
        help="Chronos prediction_length; if 0, uses horizon_step+1",
    )
    ap.add_argument("--horizon-step", type=int, default=0)

    ap.add_argument(
        "--score-mode",
        type=str,
        choices=["jump", "shift"],
        default="jump",
        help="jump: event based on (pred - last_obs) vs threshold; shift: event based on |pred - base_pred| vs threshold",
    )

    ap.add_argument("--jump-mode", type=str, choices=["up", "down", "abs"], default="abs")
    ap.add_argument("--jump-threshold", type=float, default=0.5)

    # MASCOTS baseline knobs
    ap.add_argument("--background", type=int, default=2048)
    ap.add_argument("--surrogate-samples", type=int, default=512)
    ap.add_argument("--max-grams", type=int, default=20)
    ap.add_argument("--tries-per-gram", type=int, default=30)

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num-windows", type=int, default=100)
    ap.add_argument("--save-jsonl", type=str, default="outputs/chronos_weather_cf_abs0p5_seed7_n100.jsonl")

    args = ap.parse_args()

    data_path = _resolve(args.data)
    data_raw, columns, _dates = load_weather_matrix(data_path)
    if args.target_col not in columns:
        raise SystemExit(f"target-col not found: {args.target_col}")
    target_idx = columns.index(args.target_col)

    train_slice, _val_slice, test_slice = split_slices(len(data_raw), seq_len=args.seq_len)
    train = data_raw[train_slice]
    test = data_raw[test_slice]

    prediction_length = int(args.prediction_length) if int(args.prediction_length) > 0 else int(args.horizon_step) + 1

    # Build windows
    n_test_windows = len(test) - args.seq_len - max(1, prediction_length) + 1
    if n_test_windows <= 0:
        raise SystemExit("Not enough test data for windows")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(n_test_windows, size=int(min(args.num_windows, n_test_windows)), replace=False)

    # Background: sample windows from train (history only)
    n_train_windows = len(train) - args.seq_len + 1
    k_bg = int(min(args.background, max(0, n_train_windows)))
    if k_bg <= 0:
        raise SystemExit("Not enough train data for background")
    bg_starts = rng.choice(n_train_windows, size=k_bg, replace=False)
    background = np.empty((k_bg, args.seq_len), dtype=np.float32)
    for i, s in enumerate(bg_starts):
        background[i] = train[s : s + args.seq_len, target_idx]

    print("[chronos] loading pipeline...")
    pipeline = _select_pipeline(args.model, device_map=args.device_map)
    predictor = ChronosPredictor(pipeline=pipeline, prediction_length=prediction_length, horizon_step=args.horizon_step)

    out_jsonl = _resolve(args.save_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _build_shift_background(
        hist_units: np.ndarray,
        train_background_units: np.ndarray,
        rng_local: np.random.Generator,
        max_size: int,
    ) -> np.ndarray:
        hist_units = np.asarray(hist_units, dtype=np.float32)
        if hist_units.ndim != 1:
            raise ValueError("hist_units must be 1D")
        if max_size <= 0:
            raise ValueError("max_size must be > 0")

        seq_len = hist_units.shape[0]

        # Always include the unmodified query (guarantees class 0 in shift mode).
        blocks: list[np.ndarray] = [hist_units[None, :]]

        # Near variants: small noise around the query.
        n_near = int(min(256, max(1, max_size // 4)))
        near = hist_units[None, :] + rng_local.normal(0.0, 0.05, size=(n_near, seq_len)).astype(np.float32)
        blocks.append(near.astype(np.float32))

        # Offset variants: apply a global offset (likely to change Chronos forecast).
        n_off = int(min(256, max(1, max_size // 4)))
        offsets = rng_local.choice(np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32), size=(n_off, 1))
        off = hist_units[None, :] + offsets + rng_local.normal(0.0, 0.05, size=(n_off, seq_len)).astype(np.float32)
        blocks.append(off.astype(np.float32))

        # Mix in random train windows to add diversity.
        remaining = int(max_size - sum(b.shape[0] for b in blocks))
        if remaining > 0 and len(train_background_units) > 0:
            idx = rng_local.choice(len(train_background_units), size=min(remaining, len(train_background_units)), replace=False)
            blocks.append(np.asarray(train_background_units[idx], dtype=np.float32))

        bg = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
        if bg.shape[0] > max_size:
            bg = bg[:max_size]
        perm = rng_local.permutation(bg.shape[0])
        return bg[perm]

    successes = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, win_idx in enumerate(chosen, 1):
            win_idx = int(win_idx)
            hist = test[win_idx : win_idx + args.seq_len, target_idx].astype(np.float32)

            true_next = float(test[win_idx + args.seq_len + args.horizon_step, target_idx])

            last_obs = float(hist[-1])
            pred = float(predictor.predict_next(hist))
            delta = pred - last_obs
            true_delta = true_next - last_obs

            if args.score_mode == "jump":
                if args.jump_mode == "up":
                    base_score = float(delta - args.jump_threshold)
                elif args.jump_mode == "down":
                    base_score = float((-delta) - args.jump_threshold)
                else:
                    base_score = float(abs(delta) - args.jump_threshold)
                base_class = 1 if base_score > 0 else 0
                target_class = 1 - base_class
                score_model = JumpScoreWrapper(predictor, threshold=args.jump_threshold, mode=args.jump_mode)
                background_for_fit = background
            else:
                # shift: base window is always class 0 since |pred - pred| - thr = -thr.
                base_score = float(-abs(args.jump_threshold))
                base_class = 0
                target_class = 1
                score_model = PredictionShiftScoreWrapper(predictor, base_pred=pred, threshold=args.jump_threshold)
                background_for_fit = _build_shift_background(hist, background, rng, max_size=int(args.background))

            explainer = MascotsExplainer(score_model, n_segments=8, alphabet_size=5, ngram=3)

            try:
                explainer.fit(
                    background_for_fit,
                    sample_size=min(args.surrogate_samples, len(background_for_fit)),
                    random_state=args.seed,
                )
            except ValueError as e:
                # In shift mode, a purely global background can collapse to a single class;
                # keep the sweep running instead of crashing.
                if "at least 2 classes" in str(e):
                    result = {
                        "win_idx": int(win_idx),
                        "model": args.model,
                        "device_map": args.device_map,
                        "target_col": args.target_col,
                        "seq_len": int(args.seq_len),
                        "prediction_length": int(prediction_length),
                        "horizon_step": int(args.horizon_step),
                        "score_mode": args.score_mode,
                        "jump_mode": args.jump_mode,
                        "jump_threshold": float(args.jump_threshold),
                        "last_obs_degC": float(last_obs),
                        "base_pred_degC": float(pred),
                        "base_delta_degC": float(delta),
                        "true_next_degC": float(true_next),
                        "true_delta_degC": float(true_delta),
                        "base_score": float(base_score),
                        "base_class": int(base_class),
                        "success": False,
                        "error": "surrogate_single_class",
                        "error_message": str(e),
                        "query_ts_degC": hist.astype(np.float32).tolist(),
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    if i % 10 == 0:
                        print(f"[progress] {i}/{len(chosen)} windows, successes={successes}")
                    continue
                raise

            cf_ts, cf_score, details = explainer.explain(
                hist,
                target_class=target_class,
                max_harmful_grams=args.max_grams,
                tries_per_gram=args.tries_per_gram,
                random_state=args.seed,
                return_details=True,
            )

            result: dict = {
                "win_idx": int(win_idx),
                "model": args.model,
                "device_map": args.device_map,
                "target_col": args.target_col,
                "seq_len": int(args.seq_len),
                "prediction_length": int(prediction_length),
                "horizon_step": int(args.horizon_step),
                "score_mode": args.score_mode,
                "jump_mode": args.jump_mode,
                "jump_threshold": float(args.jump_threshold),
                "last_obs_degC": float(last_obs),
                "base_pred_degC": float(pred),
                "base_delta_degC": float(delta),
                "true_next_degC": float(true_next),
                "true_delta_degC": float(true_delta),
                "base_score": float(base_score),
                "base_class": int(base_class),
                "success": bool(cf_ts is not None),
                "details": details,
                "query_ts_degC": hist.astype(np.float32).tolist(),
            }

            if cf_ts is not None:
                cf_ts = np.asarray(cf_ts, dtype=np.float32)
                cf_pred = float(predictor.predict_next(cf_ts))
                cf_delta = cf_pred - last_obs
                pred_shift = float(cf_pred - pred)
                result.update(
                    {
                        "cf_pred_degC": float(cf_pred),
                        "cf_delta_degC": float(cf_delta),
                        "pred_shift_degC": float(pred_shift),
                        "pred_shift_abs_degC": float(abs(pred_shift)),
                        "cf_score": float(cf_score) if cf_score is not None else None,
                        "cf_ts_degC": cf_ts.tolist(),
                        "delta_ts_degC": (cf_ts - hist).tolist(),
                    }
                )

                d = (cf_ts - hist).astype(np.float32)
                eps = 1e-6
                denom = int(d.size) if d.size else 0
                sparsity_count = int(np.sum(np.abs(d) > eps))
                result.update(
                    {
                        "proximity_l1": float(np.mean(np.abs(d))) if denom else None,
                        "proximity_l2": float(np.sqrt(np.mean(d * d))) if denom else None,
                        "sparsity_count": sparsity_count,
                        "sparsity_ratio": float(sparsity_count / denom) if denom else None,
                    }
                )

                successes += 1

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if i % 10 == 0:
                print(f"[progress] {i}/{len(chosen)} windows, successes={successes}")

    print("\n--- Summary ---")
    print(f"attempted={len(chosen)} successes={successes} rate={successes/len(chosen):.1%}")
    print(f"saved_jsonl={out_jsonl}")


if __name__ == "__main__":
    main()
