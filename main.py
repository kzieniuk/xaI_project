import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.model import ForecastingModel

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
    THRESHOLD_LOG_RET = 0.005 # Using 0.5% threshold for better stats
    
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

    # --- 6. MASCOTS Counterfactuals (Top 3 High & Low) ---
    print("\n--- MASCOTS Analysis (Surrogate-Guided BoRF) ---")
    
    # 1. Prepare Training Windows for Surrogate
    print("Preparing background data for MASCOTS surrogate...")
    train_vals = train_df['y'].values
    # Create sliding windows from training data (input_size=30)
    input_size = 30
    training_windows = []
    # Take a random sample or just stride to avoid excessive memory
    # Last 2000 points, stride 1
    recent_train = train_vals[-3000:]
    for i in range(len(recent_train) - input_size):
        training_windows.append(recent_train[i:i+input_size])
    training_windows = np.array(training_windows)
    
    # 2. Init and Fit Explainer
    from src.mascots import MascotsExplainer
    explainer = MascotsExplainer(model, n_segments=5, alphabet_size=5, ngram=2)
    explainer.fit(training_windows, sample_size=1024)
    
    # 3. Analyze Extremes (3 Strongest Highs, 3 Strongest Lows)
    print("\nSelect top 3 strongest predictions for each class...")
    cv_sorted = cv_df.sort_values('iTransformer')
    
    # Class 0 (Low/Negative): Lowest values
    strongest_class_0 = cv_sorted.head(3).copy()
    strongest_class_0['label'] = 'Class 0 (Strong Negative)'
    
    # Class 1 (High/Positive): Highest values
    strongest_class_1 = cv_sorted.tail(3).copy()
    strongest_class_1['label'] = 'Class 1 (Strong Positive)'
    
    # Combine (process Class 1 first maybe?)
    extreme_cases = pd.concat([strongest_class_1, strongest_class_0])
    
    for idx, row in extreme_cases.iterrows():
        ticker = row['unique_id']
        cutoff_date = row['cutoff']
        pred_val = row['iTransformer']
        label = row['label']

        
        print(f"\nAnalyzing {ticker} at {cutoff_date} (Pred: {pred_val:.4f})")
        
        # Get query TS
        ticker_df = df[df['unique_id'] == ticker]
        mask = ticker_df['ds'] <= cutoff_date
        history_window = ticker_df[mask].tail(30)
        
        if len(history_window) < 30: 
            continue
        query_ts = history_window['y'].values
        
        # Target Class: Flip Sign
        # If Pred > 0 (Class 1) -> Target 0
        # If Pred <= 0 (Class 0) -> Target 1
        target_class = 0 if pred_val > 0 else 1
        
        cf_ts, cf_pred = explainer.explain(query_ts, target_class=target_class)
        
        if cf_ts is not None:
            # Plot
            plt.figure(figsize=(10, 6))
            x_range = range(len(query_ts))
            plt.plot(x_range, query_ts, label='Original', marker='o', alpha=0.7)
            plt.plot(x_range, cf_ts, label='Counterfactual', linestyle='--', marker='x', alpha=0.7)
            
            plt.axhline(pred_val, color='blue', linestyle=':', label=f'Orig: {pred_val:.4f}')
            plt.axhline(cf_pred, color='green', linestyle=':', label=f'CF: {cf_pred:.4f}')
            
            date_str = str(row['ds']).split()[0]
            plt.title(f"MASCOTS: {ticker} on {date_str}\n{pred_val:.4f} -> {cf_pred:.4f}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = f"mascots_borf_{ticker}_{date_str}.png"
            plt.savefig(save_path)
            print(f"  Saved plot to {save_path}")
            plt.close()

if __name__ == "__main__":
    main()
