import argparse
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.mascots import MascotsExplainer
from src.mascots_paper import MascotsPaperExplainer
from src.xai import TimeSHAP


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def load_weather_matrix(path: Path, target: str = "OT") -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load weather.csv similarly to upstream Dataset_Custom.

    Returns:
    - data_raw: (T, N) raw float matrix (NOT scaled), N excludes date
    - columns: list of N column names
    - dates: (T,) numpy datetime64
    """
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"]).to_numpy()

    cols = list(df.columns)
    cols.remove(target)
    cols.remove("date")
    df = df[["date"] + cols + [target]]

    columns = list(df.columns[1:])
    data_raw = df[columns].to_numpy(dtype=np.float32)
    return data_raw, columns, dates


def split_and_scale_custom(data_raw: np.ndarray, seq_len: int) -> tuple[np.ndarray, tuple[slice, slice, slice], "sklearn.preprocessing.StandardScaler"]:
    """Match upstream Dataset_Custom split + StandardScaler fit."""
    from sklearn.preprocessing import StandardScaler

    n = len(data_raw)
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_val = n - num_train - num_test

    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_val, n]

    train_slice = slice(border1s[0], border2s[0])
    val_slice = slice(border1s[1], border2s[1])
    test_slice = slice(border1s[2], border2s[2])

    scaler = StandardScaler()
    scaler.fit(data_raw[train_slice])
    data_scaled = scaler.transform(data_raw).astype(np.float32)

    return data_scaled, (train_slice, val_slice, test_slice), scaler


def time_features_for_dates(dates: np.ndarray, freq: str) -> np.ndarray:
    """Use upstream time_features implementation."""
    import sys

    repo_root = Path(__file__).resolve().parent / "external" / "iTransformer"
    sys.path.insert(0, str(repo_root))
    from utils.timefeatures import time_features

    feats = time_features(pd.to_datetime(dates), freq=freq)  # (d_inp, L)
    return feats.transpose(1, 0).astype(np.float32)  # (L, d_inp)


class WeatherITransformerPredictor:
    """Wrapper compatible with TimeSHAP/MASCOTS: returns scalar score."""

    def __init__(
        self,
        checkpoint: Path,
        columns: list[str],
        seq_len: int,
        label_len: int,
        pred_len: int,
        freq: str,
        x_ref: np.ndarray,
        x_mark_ref: np.ndarray,
        target_col: str,
        horizon_step: int,
        device: str,
        d_model: int = 128,
        d_ff: int = 128,
        e_layers: int = 2,
        n_heads: int = 8,
    ):
        if target_col not in columns:
            raise ValueError(f"target_col not in columns: {target_col}")

        self.columns = columns
        self.target_idx = columns.index(target_col)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.horizon_step = horizon_step
        self.device = torch.device(device)

        import sys

        repo_root = Path(__file__).resolve().parent / "external" / "iTransformer"
        sys.path.insert(0, str(repo_root))
        from model.iTransformer import Model as UpstreamITransformer

        configs = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            output_attention=False,
            use_norm=True,
            embed="timeF",
            freq=freq,
            dropout=0.1,
            class_strategy="projection",
            factor=1,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            activation="gelu",
        )

        self.model = UpstreamITransformer(configs).float().to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.x_ref = torch.tensor(x_ref, dtype=torch.float32, device=self.device)
        self.x_mark_ref = torch.tensor(x_mark_ref, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _predict(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        last_label = x_enc[:, -self.label_len :, :]
        zeros = torch.zeros((x_enc.size(0), self.pred_len, x_enc.size(2)), device=x_enc.device, dtype=x_enc.dtype)
        dec_inp = torch.cat([last_label, zeros], dim=1)
        out = self.model(x_enc, x_mark_enc, dec_inp, None)
        return out

    def predict_from_array(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (self.seq_len,):
            raise ValueError(f"values must be ({self.seq_len},)")

        x = self.x_ref.clone()
        x[:, self.target_idx] = torch.tensor(values, device=self.device)
        out = self._predict(x.unsqueeze(0), self.x_mark_ref.unsqueeze(0))
        return float(out[0, self.horizon_step, self.target_idx].detach().cpu().item())

    def predict_batch(self, batch_values: np.ndarray) -> np.ndarray:
        batch_values = np.asarray(batch_values, dtype=np.float32)
        if batch_values.ndim != 2 or batch_values.shape[1] != self.seq_len:
            raise ValueError("batch_values must be (B, seq_len)")

        b = batch_values.shape[0]
        x = self.x_ref.unsqueeze(0).repeat(b, 1, 1)
        x[:, :, self.target_idx] = torch.tensor(batch_values, device=self.device)

        x_mark = self.x_mark_ref.unsqueeze(0).repeat(b, 1, 1)
        out = self._predict(x, x_mark)
        y = out[:, self.horizon_step, self.target_idx]
        return y.detach().cpu().numpy().astype(np.float32)


class WeatherITransformerPredictorMulti:
    """Like WeatherITransformerPredictor but allows overriding multiple input columns."""

    def __init__(
        self,
        checkpoint: Path,
        columns: list[str],
        seq_len: int,
        label_len: int,
        pred_len: int,
        freq: str,
        x_ref: np.ndarray,
        x_mark_ref: np.ndarray,
        output_col: str,
        horizon_step: int,
        device: str,
        d_model: int = 128,
        d_ff: int = 128,
        e_layers: int = 2,
        n_heads: int = 8,
    ):
        if output_col not in columns:
            raise ValueError(f"output_col not in columns: {output_col}")

        self.columns = columns
        self.output_idx = columns.index(output_col)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.horizon_step = horizon_step
        self.device = torch.device(device)

        import sys

        repo_root = Path(__file__).resolve().parent / "external" / "iTransformer"
        sys.path.insert(0, str(repo_root))
        from model.iTransformer import Model as UpstreamITransformer

        configs = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            output_attention=False,
            use_norm=True,
            embed="timeF",
            freq=freq,
            dropout=0.1,
            class_strategy="projection",
            factor=1,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            activation="gelu",
        )

        self.model = UpstreamITransformer(configs).float().to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.x_ref = torch.tensor(x_ref, dtype=torch.float32, device=self.device)
        self.x_mark_ref = torch.tensor(x_mark_ref, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _predict(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        last_label = x_enc[:, -self.label_len :, :]
        zeros = torch.zeros((x_enc.size(0), self.pred_len, x_enc.size(2)), device=x_enc.device, dtype=x_enc.dtype)
        dec_inp = torch.cat([last_label, zeros], dim=1)
        out = self.model(x_enc, x_mark_enc, dec_inp, None)
        return out

    def predict_from_selected(self, selected_cols: list[str], selected_values: np.ndarray) -> float:
        """selected_values: (len(selected_cols), seq_len). Returns output_col prediction (scaled)."""
        selected_values = np.asarray(selected_values, dtype=np.float32)
        if selected_values.ndim != 2 or selected_values.shape[1] != self.seq_len:
            raise ValueError("selected_values must be (d_sel, seq_len)")
        if len(selected_cols) != selected_values.shape[0]:
            raise ValueError("selected_cols length mismatch")

        x = self.x_ref.clone()
        for i, col in enumerate(selected_cols):
            if col not in self.columns:
                raise ValueError(f"col not in columns: {col}")
            idx = self.columns.index(col)
            x[:, idx] = torch.tensor(selected_values[i], device=self.device)

        out = self._predict(x.unsqueeze(0), self.x_mark_ref.unsqueeze(0))
        return float(out[0, self.horizon_step, self.output_idx].detach().cpu().item())

    def predict_batch_from_selected(self, selected_cols: list[str], batch_selected: np.ndarray) -> np.ndarray:
        """batch_selected: (B, d_sel, seq_len)."""
        batch_selected = np.asarray(batch_selected, dtype=np.float32)
        if batch_selected.ndim != 3 or batch_selected.shape[2] != self.seq_len:
            raise ValueError("batch_selected must be (B, d_sel, seq_len)")
        if batch_selected.shape[1] != len(selected_cols):
            raise ValueError("selected_cols length mismatch")

        b = batch_selected.shape[0]
        x = self.x_ref.unsqueeze(0).repeat(b, 1, 1)
        for i, col in enumerate(selected_cols):
            idx = self.columns.index(col)
            x[:, :, idx] = torch.tensor(batch_selected[:, i, :], device=self.device)

        x_mark = self.x_mark_ref.unsqueeze(0).repeat(b, 1, 1)
        out = self._predict(x, x_mark)
        y = out[:, self.horizon_step, self.output_idx]
        return y.detach().cpu().numpy().astype(np.float32)


class JumpScoreWrapper:
    """Turns a next-step prediction into a score = (pred - last_obs) - threshold."""

    def __init__(self, predictor: WeatherITransformerPredictor, last_obs: float, threshold: float):
        self.predictor = predictor
        self.last_obs = float(last_obs)
        self.threshold = float(threshold)

    def predict_from_array(self, values: np.ndarray) -> float:
        pred = self.predictor.predict_from_array(values)
        return float((pred - self.last_obs) - self.threshold)

    def predict_batch(self, batch_values: np.ndarray) -> np.ndarray:
        pred = self.predictor.predict_batch(batch_values)
        return (pred - self.last_obs - self.threshold).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual for temperature jump (Weather, iTransformer upstream)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="external/iTransformer/checkpoints/weather_96_1_smoke_iTransformer_custom_M_ft96_sl48_ll1_pl128_dm8_nh2_el1_dl128_df1_fctimeF_ebTrue_dtSmoke_projection_0/checkpoint.pth",
    )
    parser.add_argument("--data", type=str, default="iTransformer_datasets/weather/weather.csv")
    parser.add_argument("--seq-len", type=int, default=96, help="History length (96 * 10min = 16h)")
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=1, help="1 step = 10 minutes")
    parser.add_argument("--horizon-step", type=int, default=0)
    parser.add_argument("--freq", type=str, default="t")
    parser.add_argument("--target-col", type=str, default="T (degC)")
    parser.add_argument(
        "--cf-cols",
        type=str,
        default="T (degC),rh (%),p (mbar)",
        help="Comma-separated list of input columns allowed to change (multivariate CF).",
    )
    parser.add_argument("--jump-threshold-degc", type=float, default=1.0, help="Define 'dramatic' 10-min temperature jump")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Must match the upstream checkpoint architecture
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)

    parser.add_argument("--window", type=int, default=-1, help="Which window in test split to explain (-1 = last)")
    parser.add_argument("--background", type=int, default=2048)
    parser.add_argument("--surrogate-samples", type=int, default=1024)
    parser.add_argument("--tries-per-gram", type=int, default=30)
    parser.add_argument("--max-grams", type=int, default=40)

    parser.add_argument("--num-windows", type=int, default=0, help="If >0, sample this many random test windows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-jsonl",
        type=str,
        default="",
        help="If set, append per-window results as JSON Lines (good for qualitative analysis)",
    )
    parser.add_argument(
        "--max-shap",
        type=int,
        default=0,
        help="In batch mode, compute SHAP only for up to this many successful CFs (0 = never)",
    )

    parser.add_argument(
        "--print-all-changes",
        action="store_true",
        help="On successful CF, print all per-timestep changes (input->CF->Δ) in ambient space (degC)",
    )

    parser.add_argument(
        "--use-paper-mascots",
        action="store_true",
        help="Use paper-aligned MASCOTS (Algorithm 1/2 style) over selected columns (multivariate)",
    )
    parser.add_argument("--borf-w", type=int, default=24, help="BoRF receptive field length w")
    parser.add_argument("--borf-paa", type=int, default=8, help="PAA segments per receptive field")
    parser.add_argument("--borf-stride", type=int, default=1)
    parser.add_argument("--lambda-penalty", type=float, default=0.5, help="lambda penalty for pattern similarity")
    parser.add_argument("--max-iters", type=int, default=12, help="Max MASCOTS iterations")

    parser.add_argument("--run-shap", action="store_true", help="Compute SHAP for the score model on this window")
    parser.add_argument("--shap-kmeans", type=int, default=20, help="Background summarization clusters")
    parser.add_argument("--shap-nsamples", type=int, default=120, help="Kernel SHAP nsamples (speed/quality tradeoff)")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Where to save SHAP plots")

    args = parser.parse_args()

    ckpt = _resolve(args.checkpoint)
    data_path = _resolve(args.data)

    data_raw, columns, dates = load_weather_matrix(data_path)
    data_scaled, (train_slice, _, test_slice), scaler = split_and_scale_custom(data_raw, seq_len=args.seq_len)

    if args.target_col not in columns:
        raise ValueError(f"Column not found. Available include: {columns[:10]} ...")

    target_idx = columns.index(args.target_col)

    cf_cols = [c.strip() for c in args.cf_cols.split(",") if c.strip()]
    for c in cf_cols:
        if c not in columns:
            raise ValueError(f"cf col not found: {c}")
    # Convert degC jump threshold into scaled units
    temp_std = float(scaler.scale_[target_idx])
    threshold_scaled = float(args.jump_threshold_degc / temp_std)

    # Build a window from test
    test_scaled = data_scaled[test_slice]
    test_dates = dates[test_slice]

    n_windows = len(test_scaled) - args.seq_len - args.pred_len + 1
    if n_windows <= 0:
        raise RuntimeError("Not enough test data for a window")

    # Background windows for selected columns from train (same for all windows)
    rng_bg = np.random.default_rng(args.seed)
    train_block = data_scaled[train_slice]
    n_train = len(train_block)
    n_train_windows = n_train - args.seq_len + 1
    if n_train_windows <= 0:
        raise RuntimeError("Not enough train data for background windows")
    k_bg = int(min(args.background, n_train_windows))
    starts = rng_bg.choice(n_train_windows, size=k_bg, replace=False)
    cf_idxs = [columns.index(c) for c in cf_cols]
    background_mv = np.empty((k_bg, len(cf_cols), args.seq_len), dtype=np.float32)
    for i, s in enumerate(starts):
        block = train_block[s : s + args.seq_len]
        # (seq_len, N) -> (d_sel, seq_len)
        background_mv[i] = block[:, cf_idxs].T
    # For baseline univariate MASCOTS we still need 1D background (temperature)
    background_temp = background_mv[:, cf_cols.index(args.target_col), :]

    # Predictor can be reused; we will update its context tensors per window
    # (x_ref and x_mark_ref are only used as templates for other channels/time features)
    init_win = 0
    x_ref0 = test_scaled[init_win : init_win + args.seq_len]
    x_mark0 = time_features_for_dates(test_dates[init_win : init_win + args.seq_len], freq=args.freq)
    predictor = WeatherITransformerPredictor(
        checkpoint=ckpt,
        columns=columns,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        freq=args.freq,
        x_ref=x_ref0,
        x_mark_ref=x_mark0,
        target_col=args.target_col,
        horizon_step=args.horizon_step,
        device=args.device,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
    )

    predictor_mv = WeatherITransformerPredictorMulti(
        checkpoint=ckpt,
        columns=columns,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        freq=args.freq,
        x_ref=x_ref0,
        x_mark_ref=x_mark0,
        output_col=args.target_col,
        horizon_step=args.horizon_step,
        device=args.device,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
    )

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = _resolve(args.save_jsonl) if args.save_jsonl else None
    jsonl_f = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_f = jsonl_path.open("a", encoding="utf-8")

    def run_one_window(win_idx: int, run_shap_now: bool) -> dict:
        x_ref = test_scaled[win_idx : win_idx + args.seq_len]
        x_mark_ref = time_features_for_dates(test_dates[win_idx : win_idx + args.seq_len], freq=args.freq)
        predictor.x_ref = torch.tensor(x_ref, dtype=torch.float32, device=predictor.device)
        predictor.x_mark_ref = torch.tensor(x_mark_ref, dtype=torch.float32, device=predictor.device)
        predictor_mv.x_ref = predictor.x_ref
        predictor_mv.x_mark_ref = predictor.x_mark_ref

        query = x_ref[:, target_idx].astype(np.float32)
        query_mv = x_ref[:, cf_idxs].T.astype(np.float32)  # (d_sel, seq_len)
        last_obs = float(query[-1])
        if args.use_paper_mascots:
            # blackbox operates on multivariate selected inputs
            def bb_one(x_sel: np.ndarray) -> float:
                pred = predictor_mv.predict_from_selected(cf_cols, x_sel)
                return float((pred - last_obs) - threshold_scaled)

            def bb_batch(x_sel_batch: np.ndarray) -> np.ndarray:
                pred = predictor_mv.predict_batch_from_selected(cf_cols, x_sel_batch)
                return (pred - last_obs - threshold_scaled).astype(np.float32)
        else:
            jump_model = JumpScoreWrapper(predictor, last_obs=last_obs, threshold=threshold_scaled)

        if args.use_paper_mascots:
            base_score = float(bb_one(query_mv))
            base_pred = float(predictor_mv.predict_from_selected(cf_cols, query_mv))
        else:
            base_score = float(jump_model.predict_from_array(query))
            base_pred = float(predictor.predict_from_array(query))

        # Convert to degC for display
        mean = float(scaler.mean_[target_idx])
        def inv(x_scaled: float) -> float:
            return float(x_scaled * temp_std + mean)

        query_degC = (query.astype(np.float32) * temp_std + mean).astype(np.float32)

        last_obs_degC = inv(last_obs)
        base_pred_degC = inv(base_pred)
        base_delta_degC = base_pred_degC - last_obs_degC

        print("--- Setup ---")
        print(f"Target: {args.target_col} (index {target_idx})")
        print(f"Window idx (test): {win_idx}/{n_windows-1}")
        print(f"Last observed T: {last_obs_degC:.3f} °C")
        print(f"Predicted next-step T: {base_pred_degC:.3f} °C")
        print(f"Predicted delta: {base_delta_degC:.3f} °C")
        print(f"Jump threshold: {args.jump_threshold_degc:.3f} °C")
        print(f"Score = (delta - threshold): {(base_delta_degC-args.jump_threshold_degc):.3f} °C")
        print(f"Score (scaled): {base_score:.4f} -> class {1 if base_score>0 else 0}")

        print("\n--- Counterfactual (MASCOTS) ---")
        target_class = 1 if base_score <= 0 else 0
        if args.use_paper_mascots:
            explainer_p = MascotsPaperExplainer(
                blackbox_predict_one=bb_one,
                blackbox_predict_batch=bb_batch,
                w=args.borf_w,
                n_paa=args.borf_paa,
                alphabet_size=5,
                stride=args.borf_stride,
                lam=args.lambda_penalty,
                max_iters=args.max_iters,
                random_state=args.seed,
            )
            explainer_p.fit(background_mv, sample_size=min(args.surrogate_samples, len(background_mv)))
            cf_mv, cf_score, det = explainer_p.explain(query_mv, target_class=target_class)
            if cf_mv is None:
                cf_ts = None
                details = None
            else:
                # Extract temperature channel from multivariate selection
                cf_ts = cf_mv[cf_cols.index(args.target_col)].astype(np.float32)
                # Provide paper-style details for ambient analysis
                details = {
                    "paper": True,
                    "cf_cols": cf_cols,
                    "applied": det.applied if det is not None else [],
                    "iterations": det.iterations if det is not None else None,
                }
                # Persist full multivariate CF (scaled) for downstream analysis
                result_cf_mv_scaled = cf_mv.astype(np.float32)
        else:
            explainer = MascotsExplainer(jump_model, n_segments=8, alphabet_size=5, ngram=3)
            explainer.fit(background_temp, sample_size=min(args.surrogate_samples, len(background_temp)), random_state=args.seed)
            cf_ts, cf_score, details = explainer.explain(
                query,
                target_class=target_class,
                max_harmful_grams=args.max_grams,
                tries_per_gram=args.tries_per_gram,
                random_state=args.seed,
                return_details=True,
            )

        result: dict = {
            "win_idx": int(win_idx),
            "base_score_scaled": float(base_score),
            "base_pred_scaled": float(base_pred),
            "last_obs_scaled": float(last_obs),
            "last_obs_degC": float(last_obs_degC),
            "base_pred_degC": float(base_pred_degC),
            "base_delta_degC": float(base_delta_degC),
            "threshold_degC": float(args.jump_threshold_degc),
            "threshold_scaled": float(threshold_scaled),
            "target_col": args.target_col,
            "success": bool(cf_ts is not None),
            "details": details,
            # Store the original window in both spaces for downstream qualitative analysis
            "query_ts_scaled": query.astype(np.float32).tolist(),
            "query_ts_degC": query_degC.astype(np.float32).tolist(),
        }

        # Also store multivariate selected columns (scaled + original units) when enabled
        if args.use_paper_mascots:
            q_mv_scaled = query_mv.astype(np.float32)
            result["query_mv_scaled"] = q_mv_scaled.tolist()
            result["cf_cols"] = cf_cols
            # original units per column (using global scaler)
            q_mv_units = []
            for i, col in enumerate(cf_cols):
                idx = columns.index(col)
                std = float(scaler.scale_[idx])
                mu = float(scaler.mean_[idx])
                q_mv_units.append((q_mv_scaled[i] * std + mu).astype(np.float32).tolist())
            result["query_mv_units"] = q_mv_units

            # If paper explainer produced a full multivariate CF, store it + deltas
            if "result_cf_mv_scaled" in locals():
                cf_mv_scaled = result_cf_mv_scaled
                result["cf_mv_scaled"] = cf_mv_scaled.tolist()
                cf_mv_units = []
                delta_mv_units = []
                for i, col in enumerate(cf_cols):
                    idx = columns.index(col)
                    std = float(scaler.scale_[idx])
                    mu = float(scaler.mean_[idx])
                    cf_u = (cf_mv_scaled[i] * std + mu).astype(np.float32)
                    q_u = np.asarray(q_mv_units[i], dtype=np.float32)
                    cf_mv_units.append(cf_u.tolist())
                    delta_mv_units.append((cf_u - q_u).tolist())
                result["cf_mv_units"] = cf_mv_units
                result["delta_mv_units"] = delta_mv_units

        if cf_ts is None:
            print("No counterfactual found for this window.")
            return result

        if details is not None and details.get("paper"):
            print("\n--- Paper MASCOTS applied swaps (ambient subsequences) ---")
            for a in details.get("applied", [])[:50]:
                ch = int(a.get("channel"))
                t0 = int(a.get("t_start"))
                t1 = int(a.get("t_end"))
                col = cf_cols[ch] if ch < len(cf_cols) else f"ch{ch}"
                print(f"  iter={a.get('iter')} col={col} [{t0}:{t1}] p+={a.get('p_plus',{}).get('word')} -> p-={a.get('p_minus',{}).get('word')}")

            if args.print_all_changes and result.get("delta_mv_units") is not None:
                print("\n--- All per-timestep changes (all selected cols; original units) ---")
                for i, col in enumerate(cf_cols):
                    delta = np.asarray(result["delta_mv_units"][i], dtype=np.float32)
                    q_u = np.asarray(result["query_mv_units"][i], dtype=np.float32)
                    cf_u = np.asarray(result["cf_mv_units"][i], dtype=np.float32)
                    changed_idx = np.where(np.abs(delta) > 1e-6)[0]
                    if changed_idx.size == 0:
                        continue
                    print(f"[{col}] changed timesteps: {len(changed_idx)}")
                    for t in changed_idx.tolist():
                        print(f"  t={t:02d}: {float(q_u[t]):10.3f} -> {float(cf_u[t]):10.3f} (Δ={float(delta[t]):+10.3f})")
        elif details is not None:
            print("\n--- Ambient-space edits (°C) ---")
            for ch in details.get("changed_segments", []):
                start = int(ch["start"])
                end = int(ch["end"])
                orig_mean_scaled = float(ch["original_segment_mean"])
                cf_val_scaled = float(ch["counterfactual_value"])
                ch["original_segment_mean_degC"] = float(inv(orig_mean_scaled))
                ch["counterfactual_value_degC"] = float(inv(cf_val_scaled))
                print(
                    "  "
                    f"seg#{ch['segment_index']} [{start}:{end}] "
                    f"mean={ch['original_segment_mean_degC']:.3f}°C -> set={ch['counterfactual_value_degC']:.3f}°C"
                )

        if args.use_paper_mascots:
            cf_mv = np.asarray(result.get("cf_mv_scaled", []), dtype=np.float32)
            if cf_mv.size == 0:
                cf_mv = query_mv.copy()
                cf_mv[cf_cols.index(args.target_col)] = cf_ts
            cf_pred = float(predictor_mv.predict_from_selected(cf_cols, cf_mv))
        else:
            cf_pred = float(predictor.predict_from_array(cf_ts))
        cf_pred_degC = inv(cf_pred)
        cf_delta_degC = cf_pred_degC - last_obs_degC
        print("\n--- Result ---")
        print(f"CF predicted T: {cf_pred_degC:.3f} °C")
        print(f"CF predicted delta: {cf_delta_degC:.3f} °C")
        print(f"CF score (delta-threshold): {(cf_delta_degC-args.jump_threshold_degc):.3f} °C")

        l1 = float(np.mean(np.abs(cf_ts - query)))
        linf = float(np.max(np.abs(cf_ts - query)))
        changed = 0
        if details is not None:
            for ch in details.get("changed_segments", []):
                if ch.get("from_symbol") != ch.get("to_symbol"):
                    changed += 1

        result.update(
            {
                "cf_score_scaled": float(cf_score),
                "cf_pred_scaled": float(cf_pred),
                "cf_pred_degC": float(cf_pred_degC),
                "cf_delta_degC": float(cf_delta_degC),
                "mean_abs_change_scaled": float(l1),
                "max_abs_change_scaled": float(linf),
                "num_changed_segments": int(changed),
                "cf_ts_scaled": cf_ts.astype(np.float32).tolist(),
                "cf_ts_degC": (cf_ts.astype(np.float32) * temp_std + mean).astype(np.float32).tolist(),
            }
        )

        # Per-timestep deltas (always stored); for paper multivariate, store all selected cols.
        if args.use_paper_mascots:
            # Recompute cf_mv (selected cols) by replaying applied swaps from details is complex;
            # instead, store only temperature deltas unless we also persist cf_mv.
            # If we have paper details, we can at least store per-channel subsequence deltas.
            cf_degC = np.asarray(result["cf_ts_degC"], dtype=np.float32)
            q_degC = query_degC
            delta_degC = (cf_degC - q_degC).astype(np.float32)
            result["delta_ts_degC"] = delta_degC.tolist()
            if args.print_all_changes:
                print("\n--- All per-timestep changes (T degC) ---")
                for t in range(len(q_degC)):
                    dv = float(delta_degC[t])
                    if abs(dv) > 1e-6:
                        print(f"t={t:02d}: {float(q_degC[t]):8.3f} -> {float(cf_degC[t]):8.3f}  (Δ={dv:+8.3f})")
        else:
            cf_degC = np.asarray(result["cf_ts_degC"], dtype=np.float32)
            q_degC = query_degC
            delta_degC = (cf_degC - q_degC).astype(np.float32)
            result["delta_ts_degC"] = delta_degC.tolist()
            if args.print_all_changes:
                print("\n--- All per-timestep changes (degC) ---")
                for t in range(len(q_degC)):
                    qv = float(q_degC[t])
                    cv = float(cf_degC[t])
                    dv = float(delta_degC[t])
                    if abs(dv) > 1e-6:
                        print(f"t={t:02d}: {qv:8.3f} -> {cv:8.3f}  (Δ={dv:+8.3f})")

        if run_shap_now and args.max_shap != 0:
            if args.use_paper_mascots:
                print("\n[SHAP] Skipping SHAP in --use-paper-mascots mode (multivariate blackbox).")
                return result

            def _save_shap_plot(values: np.ndarray, shap_values: np.ndarray, base_value: float, title: str, path: Path) -> None:
                plt.figure(figsize=(12, 4))
                x = np.arange(len(values))
                plt.plot(x, values, color="black", alpha=0.35, linewidth=1)
                sc = plt.scatter(x, values, c=shap_values, cmap="coolwarm", s=45, edgecolor="none")
                plt.colorbar(sc, label="SHAP value (impact on score)")
                plt.title(f"{title} (base={base_value:.4f})")
                plt.xlabel("Time step (history)")
                plt.ylabel(f"{args.target_col} (scaled)")
                plt.tight_layout()
                plt.savefig(path)
                plt.close()

            shap_explainer = TimeSHAP(jump_model, background_temp, n_kmeans=args.shap_kmeans)
            q = query.reshape(1, -1)
            shap_values_q = shap_explainer.explainer.shap_values(q, nsamples=args.shap_nsamples)
            svq = shap_values_q[0] if isinstance(shap_values_q, list) else shap_values_q
            base_q = float(shap_explainer.explainer.expected_value)
            _save_shap_plot(
                values=query.astype(np.float32),
                shap_values=svq[0].astype(np.float32),
                base_value=base_q,
                title=f"SHAP (score) - original window={win_idx} thr={args.jump_threshold_degc}C",
                path=out_dir / f"weather_score_shap_win{win_idx}_thr{args.jump_threshold_degc:.2f}_orig.png",
            )
            c = cf_ts.reshape(1, -1)
            shap_values_cf = shap_explainer.explainer.shap_values(c, nsamples=args.shap_nsamples)
            svc = shap_values_cf[0] if isinstance(shap_values_cf, list) else shap_values_cf
            base_c = float(shap_explainer.explainer.expected_value)
            _save_shap_plot(
                values=cf_ts.astype(np.float32),
                shap_values=svc[0].astype(np.float32),
                base_value=base_c,
                title=f"SHAP (score) - counterfactual window={win_idx} thr={args.jump_threshold_degc}C",
                path=out_dir / f"weather_score_shap_win{win_idx}_thr{args.jump_threshold_degc:.2f}_cf.png",
            )

        return result

    try:
        if args.num_windows and args.num_windows > 0:
            rng = np.random.default_rng(args.seed)
            k = int(min(args.num_windows, n_windows))
            chosen = rng.choice(n_windows, size=k, replace=False)
            successes = 0
            attempted = 0
            shap_done = 0
            for win_idx in chosen:
                attempted += 1
                print("\n==============================")
                print(f"Batch window {attempted}/{k}: win_idx={int(win_idx)}")
                run_shap_now = bool(args.run_shap and (args.max_shap and shap_done < args.max_shap))
                res = run_one_window(int(win_idx), run_shap_now=run_shap_now)
                if res.get("success"):
                    successes += 1
                    if run_shap_now:
                        shap_done += 1
                if jsonl_f is not None:
                    jsonl_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    jsonl_f.flush()

            print("\n--- Batch summary ---")
            print(f"Attempted: {attempted}")
            print(f"Successes: {successes}")
            print(f"Success rate: {successes/attempted:.1%}" if attempted else "Success rate: n/a")
            if jsonl_path is not None:
                print(f"Saved JSONL: {jsonl_path}")
            print(f"Outputs dir: {out_dir}")
        else:
            win_idx = args.window if args.window >= 0 else (n_windows - 1)
            win_idx = int(np.clip(win_idx, 0, n_windows - 1))
            _ = run_one_window(win_idx, run_shap_now=bool(args.run_shap))
            print(f"Outputs dir: {out_dir}")
            if jsonl_path is not None:
                print(f"Saved JSONL: {jsonl_path}")
    finally:
        if jsonl_f is not None:
            jsonl_f.close()


if __name__ == "__main__":
    main()
