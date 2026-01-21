import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.mascots import MascotsExplainer
from src.xai import TimeSHAP


def _sliding_windows_1d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    if len(values) <= window:
        return np.empty((0, window), dtype=np.float32)
    return np.lib.stride_tricks.sliding_window_view(values, window_shape=window).astype(np.float32)


@dataclass
class SolarSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def load_solar_txt_matrix(path: Path) -> np.ndarray:
    # Fast, robust parse of comma-separated floats without headers.
    # Matches upstream Dataset_Solar logic.
    import pandas as pd

    wide = pd.read_csv(path, header=None)
    return wide.to_numpy(dtype=np.float32)


def solar_standardize_like_upstream(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, SolarSplit, "sklearn.preprocessing.StandardScaler"]:
    # Upstream Dataset_Solar uses split: train 70%, val 10%, test 20%.
    from sklearn.preprocessing import StandardScaler

    n = len(data)
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_valid = int(n * 0.1)

    # Borders follow upstream (note the -seq_len shift for val/test start)
    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_valid, n]

    scaler = StandardScaler()
    scaler.fit(data[border1s[0] : border2s[0]])
    scaled = scaler.transform(data)

    split = SolarSplit(
        train=scaled[border1s[0] : border2s[0]],
        val=scaled[border1s[1] : border2s[1]],
        test=scaled[border1s[2] : border2s[2]],
    )
    return scaled, split, scaler


class ITransformerWindowPredictor:
    """Predicts a single scalar from an upstream iTransformer checkpoint.

    We keep a full multivariate reference window X_ref (seq_len x N).
    The explainer perturbs ONLY one variate's history (length seq_len), and we
    embed that into X_ref before calling the multivariate model.

    The scalar prediction returned is: y_hat[horizon_step, target_var].
    """

    def __init__(
        self,
        checkpoint_path: Path,
        seq_len: int,
        label_len: int,
        pred_len: int,
        n_vars: int,
        target_var: int,
        horizon_step: int,
        x_ref: np.ndarray,
        device: str,
        d_model: int = 512,
        d_ff: int = 512,
        e_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        factor: int = 1,
        use_norm: bool = True,
    ):
        if x_ref.shape != (seq_len, n_vars):
            raise ValueError(f"x_ref must be shape ({seq_len},{n_vars}), got {x_ref.shape}")
        if not (0 <= target_var < n_vars):
            raise ValueError("target_var out of range")
        if not (0 <= horizon_step < pred_len):
            raise ValueError("horizon_step out of range")

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.target_var = target_var
        self.horizon_step = horizon_step

        self.device = torch.device(device)

        # Import upstream model
        import sys

        repo_root = Path(__file__).resolve().parent / "external" / "iTransformer"
        sys.path.insert(0, str(repo_root))
        from model.iTransformer import Model as UpstreamITransformer

        configs = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            output_attention=False,
            use_norm=use_norm,
            embed="timeF",
            freq="h",
            dropout=dropout,
            class_strategy="projection",
            factor=factor,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            activation=activation,
        )

        self.model = UpstreamITransformer(configs).float().to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.x_ref = torch.tensor(x_ref, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _predict_tensor(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc: [B, seq_len, N]
        last_label = x_enc[:, -self.label_len :, :]
        zeros = torch.zeros((x_enc.size(0), self.pred_len, x_enc.size(2)), device=x_enc.device, dtype=x_enc.dtype)
        dec_inp = torch.cat([last_label, zeros], dim=1)
        out = self.model(x_enc, None, dec_inp, None)  # [B, pred_len, N]
        return out

    def predict_from_array(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (self.seq_len,):
            raise ValueError(f"values must have shape ({self.seq_len},), got {values.shape}")

        x = self.x_ref.clone()  # [L, N]
        x[:, self.target_var] = torch.tensor(values, device=self.device)
        out = self._predict_tensor(x.unsqueeze(0))
        return float(out[0, self.horizon_step, self.target_var].detach().cpu().item())

    def predict_batch(self, batch_values: np.ndarray) -> np.ndarray:
        batch_values = np.asarray(batch_values, dtype=np.float32)
        if batch_values.ndim != 2 or batch_values.shape[1] != self.seq_len:
            raise ValueError(f"batch_values must have shape (B,{self.seq_len}), got {batch_values.shape}")

        b = batch_values.shape[0]
        x = self.x_ref.unsqueeze(0).repeat(b, 1, 1)  # [B, L, N]
        x[:, :, self.target_var] = torch.tensor(batch_values, device=self.device)

        out = self._predict_tensor(x)
        yhat = out[:, self.horizon_step, self.target_var]
        return yhat.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHAP + MASCOTS on an upstream iTransformer checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="external/iTransformer/checkpoints/solar_96_96_smoke_iTransformer_Solar_M_ft96_sl48_ll96_pl128_dm8_nh2_el1_dl128_df1_fctimeF_ebTrue_dtSmoke_projection_0/checkpoint.pth",
        help="Path to checkpoint.pth created by upstream run.py",
    )
    parser.add_argument("--data", type=str, default="iTransformer_datasets/Solar/solar_AL.txt", help="Path to solar_AL.txt")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--target-var", type=int, default=0, help="Which variate index (0..N-1) to explain")
    parser.add_argument("--horizon-step", type=int, default=0, help="Which horizon step (0..pred_len-1) to explain")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # XAI knobs
    parser.add_argument("--background-points", type=int, default=2000)
    parser.add_argument("--kmeans", type=int, default=20)
    parser.add_argument("--surrogate-samples", type=int, default=1024)

    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    # Allow running from arbitrary CWD
    if not ckpt.is_absolute():
        ckpt = (Path(__file__).resolve().parent / ckpt).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (Path(__file__).resolve().parent / data_path).resolve()

    data = load_solar_txt_matrix(data_path)
    n_vars = data.shape[1]
    _, split, _ = solar_standardize_like_upstream(data, seq_len=args.seq_len)

    # Build a reference window from the *test* split.
    x_ref = split.test[: args.seq_len]

    predictor = ITransformerWindowPredictor(
        checkpoint_path=ckpt,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        n_vars=n_vars,
        target_var=args.target_var,
        horizon_step=args.horizon_step,
        x_ref=x_ref,
        device=args.device,
        # Match upstream smoke config (d_model/d_ff were reduced to 128 there).
        d_model=128,
        d_ff=128,
        e_layers=2,
        n_heads=8,
        dropout=0.1,
        activation="gelu",
        factor=1,
        use_norm=True,
    )

    # Background windows from train portion, using only the target variate
    train_series = split.train[:, args.target_var]
    windows = _sliding_windows_1d(train_series, window=args.seq_len)
    if len(windows) == 0:
        raise RuntimeError("Not enough data to form background windows")

    if len(windows) > args.background_points:
        idx = np.random.default_rng(0).choice(len(windows), size=args.background_points, replace=False)
        background = windows[idx]
    else:
        background = windows

    query = x_ref[:, args.target_var].astype(np.float32)

    print("--- SHAP (TimeSHAP via KernelExplainer) ---")
    timeshap = TimeSHAP(predictor, background, n_kmeans=min(args.kmeans, len(background)))
    out_shap = f"itransformer_shap_var{args.target_var}_h{args.horizon_step}.png"
    timeshap.explain(query, plotting=True, save_path=out_shap)

    print("--- MASCOTS (Surrogate-Guided BoRF) ---")
    surrogate_train = background
    explainer = MascotsExplainer(predictor, n_segments=8, alphabet_size=5, ngram=3)
    explainer.fit(surrogate_train, sample_size=min(args.surrogate_samples, len(surrogate_train)))

    orig_pred = predictor.predict_from_array(query)
    orig_class = 1 if orig_pred > 0 else 0
    target_class = 1 - orig_class
    cf_ts, cf_pred = explainer.explain(query, target_class=target_class)

    if cf_ts is None:
        print("No counterfactual found.")
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    x = np.arange(len(query))
    plt.plot(x, query, label="Original", alpha=0.7)
    plt.plot(x, cf_ts, label="Counterfactual", linestyle="--", alpha=0.7)
    plt.axhline(orig_pred, color="blue", linestyle=":", label=f"Orig pred: {orig_pred:.4f}")
    plt.axhline(float(cf_pred), color="green", linestyle=":", label=f"CF pred: {float(cf_pred):.4f}")
    plt.title(f"iTransformer CF (var {args.target_var}, h {args.horizon_step})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_cf = f"itransformer_mascots_var{args.target_var}_h{args.horizon_step}.png"
    plt.savefig(out_cf)
    plt.close()
    print(f"Saved counterfactual plot to {out_cf}")


if __name__ == "__main__":
    main()
