
import argparse
from pathlib import Path
import json
import numpy as np

# Mock mpi4py to avoid Windows import errors in NeuralForecast/SHAP
import sys
from unittest.mock import MagicMock
mock_mpi = MagicMock()
mock_mpi.MPI.COMM_WORLD.Get_size.return_value = 1
sys.modules["mpi4py"] = mock_mpi
sys.modules["mpi4py.MPI"] = mock_mpi.MPI
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.model import ForecastingModel
from src.mascots import MascotsExplainer
from src.xai import TimeSHAP
from main import load_all_data, get_train_test_split

class JumpScoreWrapper:
    """
    Wraps the financial model to shift predictions by a threshold.
    Score = Prediction - Threshold.
    If Score > 0, then Prediction > Threshold.
    
    For negative targets (e.g., < -20bps), we can invert?
    But MascotsExplainer.fit binarizes based on > 0.
    
    If we want to explain a Negative Jump (Pred < -Thr):
    We want the "Event" to be "Pred < -Thr".
    So we want Score > 0 when Pred < -Thr.
    Score = -Thr - Pred.
    Example: Pred=-0.0025, Thr=0.0020. Score = -(-0.0020) - (-0.0025) ?? 
    Wait. -Thr = -0.0020.
    Score = -0.0020 - (-0.0025) = 0.0005 > 0. Correct.
    """
    def __init__(self, model, threshold, mode='above'):
        """
        threshold: positive magnitude (e.g., 0.0020)
        mode: 'above' (Pred > Thr) or 'below' (Pred < -Thr)
        """
        self.model = model
        self.threshold = float(threshold)
        self.mode = mode

    def predict_from_array(self, values):
        pred = self.model.predict_from_array(values)
        if self.mode == 'above':
            return pred - self.threshold
        elif self.mode == 'below':
            return -self.threshold - pred
        return pred

    def predict_batch(self, batch_values):
        pred = self.model.predict_batch(batch_values)
        if self.mode == 'above':
            return pred - self.threshold
        elif self.mode == 'below':
            return -self.threshold - pred
        return pred

def main():
    parser = argparse.ArgumentParser(description="Counterfactuals for Price Predictions (Financial MASCOTS)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Target ticker symbol")
    parser.add_argument("--threshold-bps", type=float, default=20.0, help="Threshold in basis points (e.g., 20 = 0.2%)")
    parser.add_argument("--direction", type=str, choices=['up', 'down'], default='up', help="Explain 'up' (positive) or 'down' (negative) moves")
    
    # Model params (used for training fresh model)
    parser.add_argument("--input-size", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000, help="Training steps (default: 3000, double the original 1500)")
    
    # MASCOTS params
    parser.add_argument("--n-segments", type=int, default=5)
    parser.add_argument("--alphabet-size", type=int, default=10)
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--surrogate-samples", type=int, default=1024)
    parser.add_argument("--max-harmful-grams", type=int, default=50)
    parser.add_argument("--tries-per-gram", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--out-dir", type=str, default="mascots_outputs")
    parser.add_argument("--evaluate-only", action="store_true", help="Stop after training and evaluation (MSE/Accuracy)")
    
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading data from {args.data_dir}...")
    df = load_all_data(args.data_dir)
    train_df, test_df = get_train_test_split(df)
    
    n_series = df['unique_id'].nunique()
    
    print(f"Training Forecasting Model (steps={args.max_steps})...")
    model = ForecastingModel(horizon=args.horizon, input_size=args.input_size, n_series=n_series, max_steps=args.max_steps)
    model.train(train_df)
    
    # --- Evaluation ---
    print("\n--- Model Evaluation ---")
    # Evaluate on Test Data using Rolling Window
    # We use the whole test set length for evaluation windows
    n_test_windows = len(test_df) // n_series
    print(f"Running Cross Validation on Test Set ({n_test_windows} steps)...")
    
    # Use model's cross_validation which wraps NeuralForecast's
    cv_df = model.cross_validation(df=df, n_windows=n_test_windows, step_size=1)
    
    # Filter for the target ticker if possible, or evaluate global?
    # Evaluation is usually per-series. Let's look at our target ticker specifically for clarity
    target_cv = cv_df[cv_df['unique_id'] == args.ticker].copy()
    
    if len(target_cv) == 0:
        print("Warning: No CV results for target ticker. Evaluating globally.")
        target_cv = cv_df.copy()
        
    # 1. MSE
    target_cv['squared_error'] = (target_cv['y'] - target_cv['iTransformer']) ** 2
    mse = target_cv['squared_error'].mean()
    rmse = np.sqrt(mse)
    print(f"Test MSE: {mse:.8f}")
    print(f"Test RMSE: {rmse:.8f}")
    
    # 2. Accuracy (Direction)
    threshold = 0.0
    target_cv['actual_dir'] = target_cv['y'] > threshold
    target_cv['pred_dir'] = target_cv['iTransformer'] > threshold
    accuracy = (target_cv['actual_dir'] == target_cv['pred_dir']).mean()
    print(f"Directional Accuracy (>0): {accuracy:.2%}")
    
    # 3. Accuracy (Big Moves - matching our CF threshold logic?)
    # Users might want to know accuracy specifically for the threshold they are targeting
    bps_decimal = args.threshold_bps / 10000.0
    if args.direction == 'up':
        target_cv['actual_event'] = target_cv['y'] > bps_decimal
        target_cv['pred_event'] = target_cv['iTransformer'] > bps_decimal
    else:
        target_cv['actual_event'] = target_cv['y'] < -bps_decimal
        target_cv['pred_event'] = target_cv['iTransformer'] < -bps_decimal
        
    event_acc = (target_cv['actual_event'] == target_cv['pred_event']).mean()
    event_recall = ((target_cv['actual_event'] & target_cv['pred_event']).sum()) / (target_cv['actual_event'].sum() + 1e-9)
    event_precision = ((target_cv['pred_event'] & target_cv['actual_event']).sum()) / (target_cv['pred_event'].sum() + 1e-9)
    
    print(f"Event Accuracy (Threshold {args.threshold_bps} bps): {event_acc:.2%}")
    print(f"Event Recall: {event_recall:.2%}")
    print(f"Event Precision: {event_precision:.2%}")

    # Try to plot loss?
    # Accessing internal trainer logs is tricky without explicit logger setup. 
    # Skipping loss plot to avoid crashes, focusing on Test Metrics.
    
    if args.evaluate_only:
        print("\nEvaluation complete. Stopping as requested.")
        return
    threshold_val = args.threshold_bps / 10000.0 # bps to decimal
    mode = 'above' if args.direction == 'up' else 'below'
    print(f"Targeting: {mode} {threshold_val:.4f} ({args.threshold_bps} bps)")
    
    wrapper = JumpScoreWrapper(model, threshold_val, mode=mode)
    
    # 4. Fit Explainer
    print("Fitting MASCOTS Surrogate...")
    # Prepare background (training windows)
    train_vals = train_df[train_df['unique_id'] == args.ticker]['y'].values
    # If not enough data for specific ticker, use all? No, specific behavior matters.
    if len(train_vals) < 1000:
        print("Warning: Low training data for ticker, using all series for background.")
        train_vals = train_df['y'].values
        
    training_windows = []
    # Stride to get diverse windows
    stride = 1
    # Limit history to recent relevant
    hist_limit = 5000
    recent_train = train_vals[-hist_limit:]
    
    for i in range(0, len(recent_train) - args.input_size, stride):
        training_windows.append(recent_train[i:i+args.input_size])
    training_windows = np.array(training_windows)
    
    explainer = MascotsExplainer(wrapper, n_segments=args.n_segments, alphabet_size=args.alphabet_size, ngram=args.ngram)
    explainer.fit(training_windows, sample_size=args.surrogate_samples)
    
    # 5. Find Target Windows in Test Data
    # We want windows where the EVENT happened (Score > 0)
    print(f"Scanning Test Data for {args.ticker} where Score > 0 (Event Occurred)...")
    target_series = test_df[test_df['unique_id'] == args.ticker]
    test_vals = target_series['y'].values
    test_dates = target_series['ds'].values
    
    found_windows = []
    
    for i in range(len(test_vals) - args.input_size):
        window = test_vals[i:i+args.input_size]
        pred_score = wrapper.predict_from_array(window)
        
        if pred_score > 0: # Event detected!
            # Predict actual value for display
            real_pred = model.predict_from_array(window)
            found_windows.append({
                'i': i,
                'date': test_dates[i+args.input_size-1], # Date of last input point? Or prediction target date?
                # NeuralForecast usually predicts t+1. Last input is t.
                'score': pred_score,
                'pred': real_pred,
                'window': window
            })
            
    print(f"Found {len(found_windows)} target windows.")
    
    # Sort by strongest score (most extreme events)
    found_windows.sort(key=lambda x: x['score'], reverse=True)
    
    # Explain top 5
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    for idx, item in enumerate(found_windows[:5]):
        print(f"\n--- Explaining Window {idx+1} ---")
        print(f"Date: {item['date']}")
        print(f"Prediction: {item['pred']:.4f}, Score: {item['score']:.4f}")
        
        # We want to flip the class from 1 (Event) to 0 (No Event)
        target_class = 0 
        
        cf_ts, cf_pred_score = explainer.explain(
            item['window'], 
            target_class=target_class,
            max_harmful_grams=args.max_harmful_grams,
            tries_per_gram=args.tries_per_gram,
            random_state=args.seed,
            return_details=True # Although our MascotsExplainer might ignore this if not fully updated, signature checks out
        )
        
        if isinstance(cf_ts, tuple): # Handle if explain returns more than 2 values (details)
             cf_ts = cf_ts[0]
             cf_pred_score = cf_ts[1]
             # Ignore details for now
        
        if cf_ts is not None:
            # Convert score back to real prediction
            if mode == 'above':
                cf_real_pred = cf_pred_score + threshold_val
            else:
                cf_real_pred = -threshold_val - cf_pred_score
                
            print(f"Counterfactual Found! New Pred: {cf_real_pred:.4f} (Score: {cf_pred_score:.4f})")
            
            # Plot
            plt.figure(figsize=(10, 6))
            x_range = np.arange(len(item['window'])) - len(item['window'])
            plt.plot(x_range, item['window'], label='Original', marker='o', alpha=0.7)
            plt.plot(x_range, cf_ts, label='Counterfactual', linestyle='--', marker='x', alpha=0.7)
            
            # Threshold Line
            if mode == 'above':
                thr_line = threshold_val
            else:
                thr_line = -threshold_val
            
            plt.axhline(item['pred'], color='blue', linestyle=':', label=f'Orig: {item["pred"]:.4f}')
            plt.axhline(cf_real_pred, color='green', linestyle=':', label=f'CF: {cf_real_pred:.4f}')
            plt.axhline(thr_line, color='red', linestyle='--', label=f'Threshold: {thr_line:.4f}')
            
            date_str = str(item['date']).split('T')[0]
            plt.title(f"MASCOTS: {args.ticker} on {date_str}\n{item['pred']:.4f} -> {cf_real_pred:.4f}")
            plt.xlabel("Days Before Prediction")
            plt.ylabel("Log Returns")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_name = f"{args.ticker}_{date_str}_{args.direction}_{args.threshold_bps}bps.png"
            plt.savefig(f"{args.out_dir}/{save_name}")
            print(f"Saved plot: {args.out_dir}/{save_name}")
            plt.close()

if __name__ == "__main__":
    main()
