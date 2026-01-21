import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.model import ForecastingModel
from src.xai import TimeSHAP
from src.mascots import MascotsExplainer


def load_solar_al_txt(file_path: str, n_series: int | None = None, freq: str = "10min") -> pd.DataFrame:
    """Load iTransformer Solar/solar_AL.txt into NeuralForecast long format.

    The file is a comma-separated numeric matrix of shape (T, N).
    Each column is treated as a separate time series (unique_id).

    Notes:
    - The dataset file has no timestamps, so we synthesize a DateTimeIndex.
    - For Solar, T=52560 corresponds to 1 year of 10-minute data (365*144).
    """
    wide = pd.read_csv(file_path, header=None)

    if n_series is not None:
        wide = wide.iloc[:, :n_series]

    t, n = wide.shape
    ds = pd.date_range(start="2000-01-01", periods=t, freq=freq)

    # Build long dataframe efficiently
    values = wide.to_numpy(dtype=np.float32)
    uid = np.repeat([f"solar_{i}" for i in range(n)], t)
    ds_rep = np.tile(ds.to_numpy(), n)
    y = values.T.reshape(-1)

    return pd.DataFrame({"unique_id": uid, "ds": ds_rep, "y": y})


def get_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_list: list[pd.DataFrame] = []
    test_list: list[pd.DataFrame] = []

    for _, group in df.groupby("unique_id", sort=False):
        n = len(group)
        train_size = int(n * train_ratio)
        train_list.append(group.iloc[:train_size])
        test_list.append(group.iloc[train_size:])

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    return train_df, test_df


def build_background_windows(series_values: np.ndarray, input_size: int, max_points: int = 5000) -> np.ndarray:
    tail = series_values[-max_points:]
    if len(tail) <= input_size:
        return np.empty((0, input_size), dtype=np.float32)

    windows = np.lib.stride_tricks.sliding_window_view(tail, window_shape=input_size)
    return windows.astype(np.float32)


def sample_training_windows(df: pd.DataFrame, input_size: int, sample_size: int = 1024, seed: int = 0) -> np.ndarray:
    """Sample windows across series without materializing all windows."""
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []

    groups = list(df.groupby("unique_id", sort=False))
    if not groups:
        return np.empty((0, input_size), dtype=np.float32)

    while len(windows) < sample_size:
        uid, g = groups[int(rng.integers(0, len(groups)))]
        vals = g["y"].to_numpy(dtype=np.float32)
        if len(vals) <= input_size:
            continue
        start = int(rng.integers(0, len(vals) - input_size))
        windows.append(vals[start : start + input_size])

    return np.stack(windows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solar AL analysis: model + SHAP + MASCOTS")
    parser.add_argument(
        "--data",
        type=str,
        default="iTransformer_datasets/Solar/solar_AL.txt",
        help="Path to solar_AL.txt",
    )
    parser.add_argument("--series", type=int, default=20, help="How many columns/series to use (<=137)")
    parser.add_argument("--freq", type=str, default="10min", help="Sampling frequency for synthetic timestamps")
    parser.add_argument("--input-size", type=int, default=144, help="History length used as model input")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    parser.add_argument("--eval-windows", type=int, default=1000, help="Rolling CV windows")
    parser.add_argument("--target-series", type=int, default=0, help="Which solar_<i> to explain")
    args = parser.parse_args()

    print("--- 1. Loading Solar AL data ---")
    df = load_solar_al_txt(args.data, n_series=args.series, freq=args.freq)
    n_series = df["unique_id"].nunique()
    print(f"Rows: {len(df):,} | Series: {n_series}")

    print("\n--- 2. Splitting Train/Test ---")
    train_df, test_df = get_train_test_split(df, train_ratio=0.8)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    print("\n--- 3. Training Model ---")
    model = ForecastingModel(horizon=args.horizon, input_size=args.input_size, n_series=n_series)
    model.train(train_df)

    print("\n--- 4. Rolling Window Evaluation ---")
    cv_df = model.cross_validation(df=df, n_windows=args.eval_windows, step_size=1)

    cv_df["squared_error"] = (cv_df["y"] - cv_df["iTransformer"]) ** 2
    rmse = float(np.sqrt(cv_df["squared_error"].mean()))
    print(f"Test RMSE: {rmse:.6f}")

    # Simple classification: high production vs not (based on 90th percentile of y)
    q90 = float(train_df["y"].quantile(0.9))
    cv_df["actual_high"] = cv_df["y"] > q90
    cv_df["pred_high"] = cv_df["iTransformer"] > q90
    acc = float((cv_df["actual_high"] == cv_df["pred_high"]).mean())
    print(f"High-output threshold (train 90p): {q90:.4f} | Accuracy: {acc:.2%}")

    print("\n--- 5. SHAP Analysis ---")
    target_uid = f"solar_{args.target_series}"
    train_target = train_df[train_df["unique_id"] == target_uid]

    background = build_background_windows(train_target["y"].to_numpy(dtype=np.float32), args.input_size, max_points=5000)
    if len(background) == 0:
        print("Not enough data for SHAP background.")
    else:
        n_kmeans = min(20, len(background))
        timeshap = TimeSHAP(model, background, n_kmeans=n_kmeans)

        full_target = df[df["unique_id"] == target_uid]
        query_ts = full_target["y"].to_numpy(dtype=np.float32)[-args.input_size :]

        print(f"Explaining latest prediction for {target_uid} with SHAP...")
        timeshap.explain(query_ts, plotting=True, save_path=f"shap_explanation_{target_uid}.png")

    print("\n--- 6. MASCOTS Analysis (Surrogate-Guided BoRF) ---")
    print("Preparing background windows for MASCOTS surrogate...")
    training_windows = sample_training_windows(train_df, input_size=args.input_size, sample_size=1024, seed=0)

    explainer = MascotsExplainer(model, n_segments=8, alphabet_size=5, ngram=3)
    explainer.fit(training_windows, sample_size=min(1024, len(training_windows)))

    # Choose a handful of extremes for counterfactuals (by prediction value)
    print("\nSelecting extremes for counterfactual generation...")
    cv_sorted = cv_df.sort_values("iTransformer")
    extremes = pd.concat([cv_sorted.tail(3), cv_sorted.head(3)], ignore_index=True)

    for _, row in extremes.iterrows():
        uid = row["unique_id"]
        cutoff = row["cutoff"]
        pred_val = float(row["iTransformer"])

        series_df = df[df["unique_id"] == uid]
        history = series_df[series_df["ds"] <= cutoff].tail(args.input_size)
        if len(history) < args.input_size:
            continue

        query_ts = history["y"].to_numpy(dtype=np.float32)
        target_class = 0 if pred_val > 0 else 1

        print(f"\nAnalyzing {uid} at {cutoff} (Pred: {pred_val:.4f})")
        cf_ts, cf_pred = explainer.explain(query_ts, target_class=target_class)
        if cf_ts is None:
            continue

        plt.figure(figsize=(10, 6))
        x = np.arange(len(query_ts))
        plt.plot(x, query_ts, label="Original", alpha=0.7)
        plt.plot(x, cf_ts, label="Counterfactual", linestyle="--", alpha=0.7)
        plt.axhline(pred_val, color="blue", linestyle=":", label=f"Orig: {pred_val:.4f}")
        plt.axhline(float(cf_pred), color="green", linestyle=":", label=f"CF: {float(cf_pred):.4f}")
        plt.title(f"MASCOTS: {uid} ({pred_val:.4f} -> {float(cf_pred):.4f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_cutoff = str(row["ds"]).split()[0]
        out_path = f"mascots_borf_{uid}_{safe_cutoff}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"  Saved plot to {out_path}")


if __name__ == "__main__":
    main()
