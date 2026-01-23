import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from chronos import ChronosPipeline

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from mascots import MascotsExplainer
except ImportError:
    # Fallback if run from root
    from src.mascots import MascotsExplainer

# --- 1. Dataset Loading ---
DATASET_PATHS = {
    "exchange_rate": os.path.join("external", "multivariate-time-series-data", "exchange_rate", "exchange_rate.txt.gz"),
    "electricity": os.path.join("external", "multivariate-time-series-data", "electricity", "electricity.txt.gz")
}

def load_dataset(name, n_series=None):
    path = DATASET_PATHS.get(name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {name} not found at {path}")
    
    print(f"Loading {name} from {path}...")
    # These datasets are typically comma-separated values, no header, one row per time step, columns are series
    # But usually .txt.gz in this repo (LSTNet/MTS data) are comma separated
    df = pd.read_csv(path, compression='gzip', header=None, sep=',')
    
    print(f"Loaded shape: {df.shape}")
    
    # If too many series, select a subset
    if n_series and n_series < df.shape[1]:
        print(f"Selecting first {n_series} series.")
        df = df.iloc[:, :n_series]
        
    return df.values # (T, N)

# --- 2. Chronos Wrapper for MASCOTS ---
class ChronosWrapper:
    def __init__(self, pipeline, prediction_length=1, num_samples=100, seed=42):
        self.pipeline = pipeline
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.seed = seed
        
    def predict(self, X):
        """
        X: (n_samples, input_len) numpy array of univariate time series windows.
        Returns: (n_samples, ) numpy array of point predictions.
        """
        # Ensure reproducibility for sampling
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            
        # Chronos expects a list of tensors or numpy arrays
        # X is (batch, time)
        
        # Convert to list of tensors for Chronos
        context = [torch.tensor(x) for x in X]
        
        # Predict
        forecast = self.pipeline.predict(
            context, 
            prediction_length=self.prediction_length,
            num_samples=self.num_samples,
            limit_prediction_length=False 
        )
        
        # Forecast is usually (Batch, NumSamples, Horizon)
        # We want the MEAN as the point prediction for gradients/CF search
        # Convert to numpy
        forecast_np = forecast.numpy() # (B, S, H)
        
        # Point estimate: Median or Mean? Mean is smoother for CF optimization proxies
        point_pred = np.mean(forecast_np, axis=1) # (B, H)
        
        return point_pred[:, 0] # Return horizon step 1

# --- 3. Jump Wrapper (for "Why is it high?") ---
class JumpScoreWrapper:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, x):
        """
        Returns score > 0 if event happens (pred > threshold).
        Event: y_pred > threshold
        Score: y_pred - threshold
        """
        preds = self.model.predict(x)
        return preds - self.threshold
        
    def predict_batch(self, x):
        return self.predict(x)
        
    def predict_from_array(self, x):
        # x is (Time,) -> reshape to (1, Time)
        return self.predict(x.reshape(1, -1))[0]

# --- 4. Visualization Functions ---
def visualize_ambient_space(x_query, x_cf, original_pred, cf_pred, target_threshold, out_path, ground_truth=None):
    plt.figure(figsize=(12, 6))
    
    T = len(x_query)
    
    # Plot Input History
    plt.plot(range(T), x_query, label=f'Original History', color='black', linewidth=2)
    plt.plot(range(T), x_cf, label=f'Counterfactual History', color='red', linestyle='--', linewidth=2)
    
    # Plot Predictions (at T)
    plt.scatter([T], [original_pred], color='black', marker='*', s=300, label=f'Original Pred: {original_pred:.2f}', zorder=10)
    plt.scatter([T], [cf_pred], color='red', marker='*', s=300, label=f'CF Pred: {cf_pred:.2f}', zorder=10)
    if ground_truth is not None:
        plt.scatter([T], [ground_truth], color='green', marker='*', s=300, label=f'Ground Truth: {ground_truth:.2f}', zorder=10)
    
    # Connect history to prediction (visual aid)
    plt.plot([T-1, T], [x_query[-1], original_pred], color='black', linestyle=':', alpha=0.5)
    plt.plot([T-1, T], [x_cf[-1], cf_pred], color='red', linestyle=':', alpha=0.5)
    if ground_truth is not None:
         plt.plot([T-1, T], [x_query[-1], ground_truth], color='green', linestyle=':', alpha=0.5)
    
    # Highlight differences
    diff = np.abs(x_query - x_cf)
    mask = diff > 1e-5
    if np.any(mask):
        # Find segments
        indices = np.where(mask)[0]
        # Allow some padding for visual clarity
        plt.fill_between(range(len(x_query)), x_query, x_cf, where=mask, color='orange', alpha=0.3, label='Perturbation')
        
    plt.axhline(target_threshold, color='green', linestyle=':', label=f'Threshold (<{target_threshold:.2f})')
    
    plt.title("MASCOTS Counterfactual (Ambient Space)")
    plt.xlabel("Time Step (T=Prediction)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Ambient plot to {out_path}")

def visualize_latent_space(explainer, background_windows, query_ts, cf_ts, out_path):
    """
    Visualize Spectral/BoRF space using PCA.
    """
    if not explainer.fitted:
        print("Explainer not fitted, cannot visualize latent space.")
        return

    # 1. Transform background data to BoRF vectors
    print("Transforming background data for latent visualization...")
    bg_sax = explainer.sax.transform(background_windows)
    bg_bags = explainer._borf(bg_sax)
    bg_vecs = explainer._vectorize(bg_bags, explainer.vocab)
    
    # 2. Transform Query and CF
    q_sax = explainer.sax.transform(query_ts.reshape(1, -1))
    q_bags = explainer._borf(q_sax)
    q_vec = explainer._vectorize(q_bags, explainer.vocab)
    
    cf_sax = explainer.sax.transform(cf_ts.reshape(1, -1))
    cf_bags = explainer._borf(cf_sax)
    cf_vec = explainer._vectorize(cf_bags, explainer.vocab)
    
    # 3. Combine for PCA fit
    # Use background for fit
    pca = PCA(n_components=2)
    bg_pca = pca.fit_transform(bg_vecs)
    
    # Transform Query/CF
    q_pca = pca.transform(q_vec)
    cf_pca = pca.transform(cf_vec)
    
    # 4. Get Surrogate Predictions (Coloring)
    surrogate_preds = explainer.surrogate.predict(bg_vecs)
    
    # 5. Plot
    plt.figure(figsize=(10, 8))
    
    # Plot background points
    # Class 0: Negative/Low (Blue), Class 1: Positive/High (Red)
    # Note: Surrogate trained on binary. check MascotsExplainer.fit logic (0 vs 1)
    
    plt.scatter(bg_pca[surrogate_preds==0, 0], bg_pca[surrogate_preds==0, 1], 
                c='blue', alpha=0.3, label='Class 0 (Low)', s=20)
    plt.scatter(bg_pca[surrogate_preds==1, 0], bg_pca[surrogate_preds==1, 1], 
                c='red', alpha=0.3, label='Class 1 (High)', s=20)
                
    # Plot Query (Star)
    plt.scatter(q_pca[:, 0], q_pca[:, 1], c='black', marker='*', s=200, label='Original Query', edgecolors='white')
    
    # Plot CF (X)
    plt.scatter(cf_pca[:, 0], cf_pca[:, 1], c='green', marker='X', s=200, label='Counterfactual', edgecolors='white')
    
    # Arrow
    plt.arrow(q_pca[0, 0], q_pca[0, 1], cf_pca[0, 0] - q_pca[0, 0], cf_pca[0, 1] - q_pca[0, 1], 
              color='black', width=0.002, head_width=0.02, length_includes_head=True, alpha=0.7)
              
    plt.title("MASCOTS Latent Generation Space (PCA of BoRF)\nSymbolic Representations")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(out_path)
    plt.close()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Latent plot to {out_path}")

def generate_comparisons(comparison_data, series, input_len, out_dir):
    """
    Generate plots comparing CFs from multiple models for the same window.
    comparison_data: dict[window_idx][model_name] = cf_array
    """
    comp_dir = os.path.join(out_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)
    
    print(f"Generating comparison plots for {len(comparison_data)} windows...")
    
    for idx, models_map in comparison_data.items():
        if len(models_map) < 2:
            continue # Skip if only 0 or 1 model succeeded (nothing to compare)
            
        # Reconstruct original window
        window = series[idx:idx+input_len]
        gt_idx = idx + input_len
        ground_truth = series[gt_idx]
        
        plt.figure(figsize=(12, 6))
        T = len(window)
        plt.plot(range(T), window, label='Original', color='black', linewidth=2, alpha=0.7)
        
        # Plot GT
        plt.scatter([T], [ground_truth], color='green', marker='*', s=200, label='Ground Truth', zorder=10)
        
        colors = {'amazon/chronos-t5-tiny': 'blue', 'amazon/chronos-t5-small': 'orange', 'amazon/chronos-t5-base': 'red'}
        
        for model_name, cf in models_map.items():
            color = colors.get(model_name, 'purple')
            short_name = model_name.split('-')[-1] # tiny, small, base
            plt.plot(range(T), cf, label=f'CF {short_name}', color=color, linestyle='--', linewidth=1.5)
            
        plt.title(f"Model Comparison - Window {idx}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(comp_dir, f"compare_win{idx}.png"))
        plt.close()

# --- 5. Main Logic ---

def run_sweep(args):
    import csv
    import time
    
    models = ['amazon/chronos-t5-tiny', 'amazon/chronos-t5-small', 'amazon/chronos-t5-base']
    if args.model != "sweep" and args.model != "amazon/chronos-t5-tiny": # If specific model requested
         if "chronos" in args.model:
             models = [args.model]
    
    # Allow user to override just one model for testing sweep logic
    # Actually, sweep implies multiple, but if user passes --model, maybe just that one?
    # Let's check if args.model is default.
    if args.model != "amazon/chronos-t5-tiny":
         models = [args.model]

    # Load Data ONCE
    data = load_dataset(args.dataset, n_series=args.n_series)
    T, N = data.shape
    input_len = 64
    series = data[:, 0] # Series 0
    
    # Select 100 random indices (Fixed Seed)
    np.random.seed(42)
    possible_indices = np.arange(0, T - input_len - 1)
    
    n_samples = args.n_samples
    print(f"Sweeping {n_samples} random samples...")
    sample_indices = np.random.choice(possible_indices, size=n_samples, replace=False)
    sample_indices.sort()
    
    print(f"Selected {n_samples} sample windows. Running sweep on models: {models}")
    
    # Prepare background data for fitting (common pool)
    bg_pool_indices = np.random.choice(possible_indices, size=200, replace=False)
    bg_pool = np.array([series[i:i+input_len] for i in bg_pool_indices])
    
    # Base output dir
    base_out = os.path.join(args.out_dir, args.dataset) # e.g. chronos_outputs/electricity
    os.makedirs(base_out, exist_ok=True)
    
    summary_file = os.path.join(base_out, "sweep_summary.csv")
    file_exists = os.path.exists(summary_file)
    with open(summary_file, 'a', newline='') as f: # Append mode
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "WindowIdx", "OriginalPred", "Target", "CFPred", "IsSuccess", "Cost", "Time"])

    # Dictionary to store CFs for comparison: comparison_data[idx][model] = cf
    comparison_data = {}

    for model_name in models:
        print(f"\n--- Processing Model: {model_name} ---")
        safe_model_name = model_name.replace('/', '-')
        model_out_dir = os.path.join(base_out, safe_model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        
        # Load Model
        try:
            pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
            # Use configurable num_samples (default lower for sweep?)
            # Actually use the arg directly
            wrapper = ChronosWrapper(pipeline, seed=42, num_samples=args.chronos_samples)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
        
        success_count = 0
        
        for idx in tqdm(sample_indices, desc=f"Sweeping {model_name}"):
            window = series[idx:idx+input_len]
            gt_idx = idx + input_len
            ground_truth = series[gt_idx]
            
            start_time = time.time()
            
            # Predict
            preds = wrapper.predict(window.reshape(1, -1))
            orig_pred = preds[0]
            
            # Skip if pred is trivially 0 (e.g. padding?)
            if abs(orig_pred) < 1e-4:
                continue

            target = orig_pred * args.threshold_scale
            
            # Run MASCOTS
            try:
                cf_model = JumpScoreWrapper(wrapper, target)
                explainer = MascotsExplainer(
                    cf_model,
                    n_segments=args.n_segments,
                    ngram=args.ngram,
                    alphabet_size=args.alphabet_size
                )
                
                # Use shared pool for fit for consistency and speed
                explainer.fit(bg_pool, sample_size=100)
                
                cf, res = explainer.explain(
                    window,
                    max_harmful_grams=50,
                    tries_per_gram=args.tries_per_gram, # Configurable
                    random_state=42
                )
                
                duration = time.time() - start_time
                cf_pred = 0
                is_success = False
                cost = 0
                
                if cf is not None:
                    is_success = True
                    success_count += 1
                    cf_pred = wrapper.predict(cf.reshape(1, -1))[0]
                    cost = np.mean((cf - window) ** 2)
                    
                    # Save Ambient Plot
                    plot_bname = f"ambient_win{idx}.png"
                    visualize_ambient_space(
                        window, cf, orig_pred, cf_pred, target,
                        os.path.join(model_out_dir, plot_bname),
                        ground_truth
                    )
                    
                    # Store for comparison
                    if idx not in comparison_data:
                        comparison_data[idx] = {}
                    comparison_data[idx][model_name] = cf
                    
            except Exception as e:
                print(f"Error processing window {idx}: {e}")
                cf_pred = 0
                is_success = False
                cost = 0
                duration = time.time() - start_time
            
            # Append to CSV
            with open(summary_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_name, idx, orig_pred, target, cf_pred, is_success, cost, duration])
                
        print(f"Model {model_name}: Success Rate {success_count}/100")
        
    # Generate Comparisons
    generate_comparisons(comparison_data, series, input_len, base_out)
def run_single_analysis(args):
    # Load Model
    print(f"Loading Chronos model: {args.model}")
    pipeline = ChronosPipeline.from_pretrained(
        args.model,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    
    wrapper = ChronosWrapper(pipeline, seed=42)
    
    # Load Data
    data = load_dataset(args.dataset, n_series=args.n_series)
    # data is (T, N)
    
    T, N = data.shape
    input_len = 64 # Chronos context length
    
    # Select a target series
    target_idx = 0 # Just pick the first one for now, or random?
    print(f"Analyzing Series {target_idx}")
    series = data[:, target_idx]
    
    # Prepare windows
    # We want to find EXTREME predictions
    print("Scanning for extreme predictions...")
    
    # Rolling window approach
    # We'll sample 100 windows randomly to avoid full scan slowness
    # indices = np.random.choice(range(T - input_len - 1), size=100, replace=False)
    # Actually let's scan a chunk to be deterministic
    scan_limit = min(T - input_len - 1, 500)
    indices = np.arange(0, scan_limit, 5) # Stride 5
    
    windows = []
    for i in indices:
        windows.append(series[i:i+input_len])
    X_scan = np.array(windows)
    
    print(f"Predicting batch of {len(X_scan)} windows...")
    preds = wrapper.predict(X_scan)
    
    # Find max prediction
    max_idx = np.argmax(preds)
    best_pred = preds[max_idx]
    best_window = X_scan[max_idx]
    
    print(f"Found Max Prediction: {best_pred:.4f} at window index {indices[max_idx]}")
    
    # Get Ground Truth
    gt_idx = indices[max_idx] + input_len
    ground_truth = series[gt_idx] if gt_idx < len(series) else None
    print(f"Ground Truth Value: {ground_truth:.4f}")
    
    # --- Run MASCOTS ---
    # Goal: Lower the prediction? (Why is it high?)
    # Threshold: Let's aim to reduce it by X% (configurable)
    target_threshold = best_pred * args.threshold_scale
    print(f"Goal: Generate CF to reduce prediction below {target_threshold:.4f} (Scale: {args.threshold_scale})")
    
    # Wrap for MASCOTS (Score = Threshold - Pred ? No, we want Pred < Threshold)
    # Score > 0 means Pred < Threshold?
    # No, typically decision box is decision_function(x) > 0 => Class 1.
    # Current MASCOTS assumes we want to flip the sign of the score.
    # If current score is "Pred - Threshold" (Positive), we want to make it Negative.
    # So `JumpScoreWrapper` returning `Pred - Threshold` works.
    # Original state: Pred (~High) - Threshold (~High*0.9) > 0.
    # Counterfactual state: Pred' < Threshold => Score < 0.
    
    cf_model = JumpScoreWrapper(wrapper, target_threshold)
    
    explainer = MascotsExplainer(
        cf_model,
        n_segments=args.n_segments,
        ngram=args.ngram,
        alphabet_size=args.alphabet_size
    )
    
    # Define query
    x_query = best_window
    
    # Need background data for fitting logic (surrogate)
    # We can use the windows we scanned
    explainer.fit(X_scan, sample_size=512)
    # Checking mascots.py: fit(X, y=None) -> learns BoRF
    
    print("Generating Counterfactual...")
    # Passing dynamic args
    cf, res = explainer.explain(
        x_query,
        max_harmful_grams=50,
        tries_per_gram=50,
        random_state=42,
        return_details=True
    )
    
    if cf is not None:
        print("Counterfactual Found!")
        cf_pred = wrapper.predict(cf.reshape(1, -1))[0]
        print(f"Original Pred: {best_pred:.4f}")
        print(f"CF Pred: {cf_pred:.4f}")
        print(f"Threshold: {target_threshold:.4f}")
        
        # Visualize
        os.makedirs(args.out_dir, exist_ok=True)

        # Visualize Ambient
        ambient_path = os.path.join(args.out_dir, f"chronos_viz_ambient_{args.dataset}.png")
        visualize_ambient_space(x_query, cf, best_pred, cf_pred, target_threshold, ambient_path, ground_truth)
        
        # Visualize Latent
        latent_path = os.path.join(args.out_dir, f"chronos_viz_latent_{args.dataset}.png")
        visualize_latent_space(explainer, X_scan, x_query, cf, latent_path)
        
    else:
        print("No counterfactual found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="exchange_rate", choices=["exchange_rate", "electricity"])
    parser.add_argument("--model", type=str, default="amazon/chronos-t5-tiny")
    parser.add_argument("--n-series", type=int, default=1, help="Number of series to check (takes first N)")
    parser.add_argument("--out-dir", type=str, default="chronos_outputs")
    parser.add_argument("--threshold-scale", type=float, default=0.90, help="Target threshold scale (default 0.90 for 10% reduction)")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of random samples for sweep (default 100)")
    parser.add_argument("--chronos-samples", type=int, default=20, help="Number of samples for Chronos prediction (lower = faster)")
    parser.add_argument("--tries-per-gram", type=int, default=10, help="Search candidates per gram (lower = faster)")
    
    parser.add_argument("--sweep", action="store_true", help="Run batch sweep on N random samples")
    
    # Mascots params
    parser.add_argument("--n-segments", type=int, default=8)
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--alphabet-size", type=int, default=10)
    
    args = parser.parse_args()
    
    if args.sweep:
        run_sweep(args)
    else:
        run_single_analysis(args)
