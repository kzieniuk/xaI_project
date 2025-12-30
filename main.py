import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.model import ForecastingModel
from src.mascots import MASCOTS
from src.xai import TimeSHAP

def load_all_data(data_dir):
    all_dfs = []
    csv_files = glob.glob(f"{data_dir}/*.csv")
    print(f"Found {len(csv_files)} files: {csv_files}")
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        ticker = filename.split('_')[1]
        
        df = pd.read_csv(filepath)
        if 'Date' not in df.columns and df.columns[0].startswith('Unnamed'):
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        # Ensure UTC and remove tz info
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        df = df.sort_values('Date')
        
        # Compute Log Returns: ln(P_t / P_{t-1})
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()
        
        nf_df = pd.DataFrame({
            'unique_id': ticker,
            'ds': df['Date'],
            'y': df['log_ret'] # Target is now Log Returns
        })
        all_dfs.append(nf_df)
            
    return pd.concat(all_dfs)

def get_train_test_split(df, train_ratio=0.8):
    train_list = []
    test_list = []
    
    for uid, group in df.groupby('unique_id'):
        n = len(group)
        train_size = int(n * train_ratio)
        train_list.append(group.iloc[:train_size])
        test_list.append(group.iloc[train_size:])
        
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df

def main():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("Data directory not found. Please run download_data.py first.")
        return

    print("--- 1. Loading Data (Log-Returns) ---")
    df = load_all_data(data_dir)
    n_series = df['unique_id'].nunique()
    print(f"Total rows: {len(df)}, Series: {n_series}")
    
    print("\n--- 2. Splitting Train (80%) / Test (20%) ---")
    train_df, test_df = get_train_test_split(df, train_ratio=0.8)
    print(f"Train Size: {len(train_df)}")
    print(f"Test Size: {len(test_df)}")
    
    print("\n--- 3. Training Model on Log-Returns ---")
    # Horizon=1: Predicting next day's log-return
    model = ForecastingModel(horizon=1, input_size=30, n_series=n_series)
    model.train(train_df)
    
    print("\n--- 4. Testing (Rolling Window Evaluation) ---")
    # Evaluate on the last 180 days of the test set for specific demonstration
    EVAL_WINDOW = 180
    print(f"Evaluating rolling window over last {EVAL_WINDOW} days of Test Set...")
    
    cv_df = model.cross_validation(df=df, n_windows=EVAL_WINDOW, step_size=1)
    
    # 1. MSE
    cv_df['squared_error'] = (cv_df['y'] - cv_df['iTransformer']) ** 2
    rmse = np.sqrt(cv_df['squared_error'].mean())
    print(f"Test RMSE (Log-Return scale): {rmse:.6f}")
    
    # 2. Classification Accuracy for "Big Moves"
    THRESHOLD_LOG_RET = 0.00
    
    cv_df['actual_big_move'] = cv_df['y'] > THRESHOLD_LOG_RET
    cv_df['pred_big_move'] = cv_df['iTransformer'] > THRESHOLD_LOG_RET
    
    cv_df['correct'] = cv_df['actual_big_move'] == cv_df['pred_big_move']
    accuracy = cv_df['correct'].mean()
    
    print(f"\nCondition: Daily Return > {THRESHOLD_LOG_RET*100}%")
    print(f"Accuracy: {accuracy:.2%}")
    
    tp = ((cv_df['pred_big_move']) & (cv_df['actual_big_move'])).sum()
    fp = ((cv_df['pred_big_move']) & (~cv_df['actual_big_move'])).sum()
    tn = ((~cv_df['pred_big_move']) & (~cv_df['actual_big_move'])).sum()
    fn = ((~cv_df['pred_big_move']) & (cv_df['actual_big_move'])).sum()
    
    print(f"Confusion Matrix:")
    print(f"  TP: {tp} | FP: {fp}")
    print(f"  FN: {fn} | TN: {tn}")
    
    target_ticker = "AAPL"
    aapl_cv = cv_df[cv_df['unique_id'] == target_ticker]
    
    if not aapl_cv.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(aapl_cv['ds'], aapl_cv['y'], label='Actual LogRet', alpha=0.5)
        plt.plot(aapl_cv['ds'], aapl_cv['iTransformer'], label='Pred LogRet', alpha=0.5, linestyle='--')
        plt.axhline(THRESHOLD_LOG_RET, color='red', linestyle=':', label=f'{THRESHOLD_LOG_RET*100}% Threshold')
        
        # Highlight True Positives
        hits = aapl_cv[aapl_cv['pred_big_move'] & aapl_cv['actual_big_move']]
        if not hits.empty:
            plt.scatter(hits['ds'], hits['y'], color='green', marker='^', s=80, label='TP', zorder=5)
            
        plt.title(f"{target_ticker} Test (Last {EVAL_WINDOW} Days) - Log Returns\nAcc (>{THRESHOLD_LOG_RET*100}%): {accuracy:.2%} | RMSE: {rmse:.6f}")
        plt.legend()
        plt.savefig("test_evaluation_logret.png")
        print("Saved plot to test_evaluation_logret.png")

    # --- 5. SHAP Analysis ---
    print("\n--- SHAP Analysis ---")
    # Prepare background data for SHAP (summary of training data)
    train_target = train_df[train_df['unique_id'] == target_ticker]
    
    # We need to construct windows of scale (N, 30) from the training data.
    history_vals = train_target['y'].values[-500:] # Last 500 training points
    windows = []
    # Create sliding windows
    for i in range(len(history_vals) - 30):
        windows.append(history_vals[i:i+30])
    
    if len(windows) > 0:
        background_data = np.array(windows)
        
        # Select background size: K-Means needs enough data. 
        # If very few windows, use them directly, else use KMeans
        n_kmeans = min(20, len(background_data))
        timeshap = TimeSHAP(model, background_data, n_kmeans=n_kmeans)
        
        # Explain the LAST window of the test set (most recent known data)
        target_series = df[df['unique_id'] == target_ticker]
        query_ts = target_series['y'].values[-30:]
        
        print("Explaining latest prediction with SHAP...")
        timeshap.explain(query_ts, plotting=True, save_path=f"shap_explanation_{target_ticker}.png")
    else:
        print("Not enough data for SHAP background.")

    # --- 6. MASCOTS Counterfactual ---
    print("\n--- MASCOTS Analysis ---")
    target_series = df[df['unique_id'] == target_ticker]
    query_ts = target_series['y'].values[-30:] # Last 30 days of log-returns
    last_date = target_series['ds'].iloc[-1]
    next_date = last_date + pd.Timedelta(days=1)
    
    current_pred = model.predict_from_array(query_ts)
    print(f"Current Prediction for {next_date.date()}: {current_pred:.4f}")

    # Goal: Flip the sign of prediction (approx) or reach a significant threshold
    
    target_threshold = 0.00
    if current_pred < 0:
        goal_desc = f"> {target_threshold}"
        def condition(pred): return pred > target_threshold
    else:
        goal_desc = f"< -{target_threshold}"
        def condition(pred): return pred < -target_threshold

    print(f"Goal: Find counterfactual with prediction {goal_desc}")

    mascots = MASCOTS(model)
    # Reduced max_iter for speed in demo
    cf_ts, cf_pred = mascots.generate_counterfactual(query_ts, condition, max_iter=300)
    
    if cf_ts is not None:
        print(f"Counterfactual found! Original: {current_pred:.4f} -> Counterfactual: {cf_pred:.4f}")
        
        plt.figure(figsize=(10, 6))
        x_range = range(len(query_ts))
        plt.plot(x_range, query_ts, label='Original History', marker='o', alpha=0.7)
        plt.plot(x_range, cf_ts, label='Counterfactual History', linestyle='--', marker='x', alpha=0.7)
        
        plt.axhline(current_pred, color='blue', linestyle=':', label=f'Orig Pred: {current_pred:.4f}')
        plt.axhline(cf_pred, color='green', linestyle=':', label=f'CF Pred: {cf_pred:.4f}')
        
        plt.title(f"MASCOTS: {target_ticker} for {next_date.date()}\nChange: {current_pred:.4f} -> {cf_pred:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = f"mascots_explanation_{target_ticker}.png"
        plt.savefig(save_path)
        print(f"Saved MASCOTS plot to {save_path}")
    else:
        print("No counterfactual found within iterations.")

if __name__ == "__main__":
    main()
