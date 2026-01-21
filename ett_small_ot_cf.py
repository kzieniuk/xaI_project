import argparse
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import torch

from src.mascots_paper import MascotsPaperExplainer


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def load_ett_matrix(path: Path, target: str = "OT") -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load ETT-small csv.

    Returns:
    - data_raw: (T, N) float32 matrix (no date)
    - columns: list of N column names (no date)
    - dates: (T,) datetime64
    """
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"]).to_numpy()
    columns = [c for c in df.columns if c != "date"]

    if target not in columns:
        raise ValueError(f"target not found: {target}")

    # Keep original order; ensure target exists (usually last already)
    data_raw = df[columns].to_numpy(dtype=np.float32)
    return data_raw, columns, dates


def split_and_scale(data_raw: np.ndarray, seq_len: int):
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
    import sys

    repo_root = Path(__file__).resolve().parent / "external" / "iTransformer"
    sys.path.insert(0, str(repo_root))
    from utils.timefeatures import time_features

    feats = time_features(pd.to_datetime(dates), freq=freq)
    return feats.transpose(1, 0).astype(np.float32)


class ITransformerPredictorMulti:
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
        d_model: int,
        d_ff: int,
        e_layers: int,
        n_heads: int,
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
        return self.model(x_enc, x_mark_enc, dec_inp, None)

    def predict_from_selected(self, selected_cols: list[str], selected_values: np.ndarray) -> float:
        selected_values = np.asarray(selected_values, dtype=np.float32)
        if selected_values.ndim != 2 or selected_values.shape[1] != self.seq_len:
            raise ValueError("selected_values must be (d_sel, seq_len)")
        if len(selected_cols) != selected_values.shape[0]:
            raise ValueError("selected_cols length mismatch")

        x = self.x_ref.clone()
        for i, col in enumerate(selected_cols):
            idx = self.columns.index(col)
            x[:, idx] = torch.tensor(selected_values[i], device=self.device)

        out = self._predict(x.unsqueeze(0), self.x_mark_ref.unsqueeze(0))
        return float(out[0, self.horizon_step, self.output_idx].detach().cpu().item())

    def predict_batch_from_selected(self, selected_cols: list[str], batch_selected: np.ndarray) -> np.ndarray:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-style MASCOTS CF for ETT-small (iTransformer)")
    ap.add_argument("--data", type=str, default="iTransformer_datasets/ETT-small/ETTh1.csv")
    ap.add_argument("--target-col", type=str, default="OT")
    ap.add_argument(
        "--cf-cols",
        type=str,
        default="OT,HUFL,HULL,MUFL,MULL,LUFL,LULL",
        help="Comma-separated cols allowed to change",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to iTransformer checkpoint.pth trained on this ETT variant",
    )

    # iTransformer script defaults for ETTh1_96_96
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--label-len", type=int, default=48)
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--horizon-step", type=int, default=0)
    ap.add_argument("--freq", type=str, default="h")

    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--e-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=8)

    ap.add_argument("--jump-threshold", type=float, default=1.0, help="Meaningful jump threshold in target units")

    ap.add_argument("--background", type=int, default=2048)
    ap.add_argument("--surrogate-samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--borf-w", type=int, default=24)
    ap.add_argument("--borf-paa", type=int, default=8)
    ap.add_argument("--borf-stride", type=int, default=1)
    ap.add_argument("--lambda-penalty", type=float, default=0.5)
    ap.add_argument("--max-iters", type=int, default=12)

    ap.add_argument("--window", type=int, default=-1)
    ap.add_argument("--num-windows", type=int, default=0)
    ap.add_argument("--save-jsonl", type=str, default="")
    ap.add_argument("--print-all-changes", action="store_true")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    data_path = _resolve(args.data)
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required (train ETTh1_96_96 first)")
    ckpt = _resolve(args.checkpoint)

    data_raw, columns, dates = load_ett_matrix(data_path, target=args.target_col)
    data_scaled, (train_slice, _, test_slice), scaler = split_and_scale(data_raw, seq_len=args.seq_len)

    cf_cols = [c.strip() for c in args.cf_cols.split(",") if c.strip()]
    for c in cf_cols:
        if c not in columns:
            raise ValueError(f"cf col not found: {c}")

    target_idx = columns.index(args.target_col)
    cf_idxs = [columns.index(c) for c in cf_cols]

    test_scaled = data_scaled[test_slice]
    test_dates = dates[test_slice]

    n_windows = len(test_scaled) - args.seq_len - args.pred_len + 1
    if n_windows <= 0:
        raise RuntimeError("Not enough test data for a window")

    rng = np.random.default_rng(args.seed)

    # Background multivariate windows from train
    train_block = data_scaled[train_slice]
    n_train = len(train_block)
    n_train_windows = n_train - args.seq_len + 1
    if n_train_windows <= 0:
        raise RuntimeError("Not enough train data for background")
    k_bg = int(min(args.background, n_train_windows))
    starts = rng.choice(n_train_windows, size=k_bg, replace=False)
    background_mv = np.empty((k_bg, len(cf_cols), args.seq_len), dtype=np.float32)
    for i, s in enumerate(starts):
        block = train_block[s : s + args.seq_len]
        background_mv[i] = block[:, cf_idxs].T

    # predictor templates
    init_win = 0
    x_ref0 = test_scaled[init_win : init_win + args.seq_len]
    x_mark0 = time_features_for_dates(test_dates[init_win : init_win + args.seq_len], freq=args.freq)

    predictor = ITransformerPredictorMulti(
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

    jsonl_path = _resolve(args.save_jsonl) if args.save_jsonl else None
    jsonl_f = jsonl_path.open("a", encoding="utf-8") if jsonl_path is not None else None

    def inv_col(col: str, x_scaled: np.ndarray) -> np.ndarray:
        idx = columns.index(col)
        return (x_scaled * float(scaler.scale_[idx]) + float(scaler.mean_[idx])).astype(np.float32)

    def run_one(win_idx: int) -> dict:
        x_ref = test_scaled[win_idx : win_idx + args.seq_len]
        x_mark_ref = time_features_for_dates(test_dates[win_idx : win_idx + args.seq_len], freq=args.freq)
        predictor.x_ref = torch.tensor(x_ref, dtype=torch.float32, device=predictor.device)
        predictor.x_mark_ref = torch.tensor(x_mark_ref, dtype=torch.float32, device=predictor.device)

        query_mv = x_ref[:, cf_idxs].T.astype(np.float32)
        query_target = x_ref[:, target_idx].astype(np.float32)
        last_obs = float(query_target[-1])

        # threshold in scaled units for target
        thr_scaled = float(args.jump_threshold / float(scaler.scale_[target_idx]))

        def bb_one(x_sel: np.ndarray) -> float:
            pred = predictor.predict_from_selected(cf_cols, x_sel)
            return float((pred - last_obs) - thr_scaled)

        def bb_batch(x_sel_batch: np.ndarray) -> np.ndarray:
            pred = predictor.predict_batch_from_selected(cf_cols, x_sel_batch)
            return (pred - last_obs - thr_scaled).astype(np.float32)

        base_score = float(bb_one(query_mv))
        base_pred = float(predictor.predict_from_selected(cf_cols, query_mv))

        # Units for reporting (original data units, via inverse scaling)
        last_obs_u = float(inv_col(args.target_col, np.asarray(last_obs, dtype=np.float32)))
        base_pred_u = float(inv_col(args.target_col, np.asarray(base_pred, dtype=np.float32)))
        base_delta_u = base_pred_u - last_obs_u

        target_class = 1 if base_score <= 0 else 0

        expl = MascotsPaperExplainer(
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
        expl.fit(background_mv, sample_size=min(args.surrogate_samples, len(background_mv)))

        cf_mv, cf_score, det = expl.explain(query_mv, target_class=target_class)

        result = {
            "win_idx": int(win_idx),
            "success": bool(cf_mv is not None),
            "target_col": args.target_col,
            "cf_cols": cf_cols,
            "threshold": float(args.jump_threshold),
            "threshold_scaled": float(thr_scaled),
            "last_obs_scaled": float(last_obs),
            "base_pred_scaled": float(base_pred),
            "base_score_scaled": float(base_score),
            "last_obs_units": float(last_obs_u),
            "base_pred_units": float(base_pred_u),
            "base_delta_units": float(base_delta_u),
            "query_mv_scaled": query_mv.tolist(),
            "query_mv_units": [inv_col(c, query_mv[i]).tolist() for i, c in enumerate(cf_cols)],
            "details": {
                "paper": True,
                "iterations": det.iterations if det is not None else None,
                "applied": det.applied if det is not None else [],
            },
        }

        if cf_mv is None:
            return result

        cf_mv = cf_mv.astype(np.float32)
        cf_pred = float(predictor.predict_from_selected(cf_cols, cf_mv))
        cf_pred_u = float(inv_col(args.target_col, np.asarray(cf_pred, dtype=np.float32)))
        cf_delta_u = cf_pred_u - last_obs_u

        result.update(
            {
                "cf_score_scaled": float(cf_score),
                "cf_pred_scaled": float(cf_pred),
                "cf_pred_units": float(cf_pred_u),
                "cf_delta_units": float(cf_delta_u),
                "cf_mv_scaled": cf_mv.tolist(),
                "cf_mv_units": [inv_col(c, cf_mv[i]).tolist() for i, c in enumerate(cf_cols)],
                "delta_mv_units": [(inv_col(c, cf_mv[i]) - inv_col(c, query_mv[i])).tolist() for i, c in enumerate(cf_cols)],
            }
        )

        if args.print_all_changes:
            print("\n--- All per-timestep changes (all selected cols; units) ---")
            for i, col in enumerate(cf_cols):
                q_u = np.asarray(result["query_mv_units"][i], dtype=np.float32)
                c_u = np.asarray(result["cf_mv_units"][i], dtype=np.float32)
                d_u = np.asarray(result["delta_mv_units"][i], dtype=np.float32)
                idxs = np.where(np.abs(d_u) > 1e-6)[0]
                if idxs.size == 0:
                    continue
                print(f"[{col}] changed timesteps: {len(idxs)}")
                for t in idxs.tolist():
                    print(f"  t={t:02d}: {float(q_u[t]):10.3f} -> {float(c_u[t]):10.3f} (Δ={float(d_u[t]):+10.3f})")

        return result

    try:
        if args.num_windows and args.num_windows > 0:
            chosen = rng.choice(n_windows, size=int(min(args.num_windows, n_windows)), replace=False)
            for i, win_idx in enumerate(chosen, 1):
                print(f"\n=== Window {i}/{len(chosen)}: win_idx={int(win_idx)} ===")
                res = run_one(int(win_idx))
                if jsonl_f is not None:
                    jsonl_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    jsonl_f.flush()
        else:
            win_idx = args.window if args.window >= 0 else (n_windows - 1)
            win_idx = int(np.clip(win_idx, 0, n_windows - 1))
            _ = run_one(win_idx)
    finally:
        if jsonl_f is not None:
            jsonl_f.close()


if __name__ == "__main__":
    main()
