import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TimeSHAP:
    def __init__(self, model, background_data, n_kmeans=20):
        """
        model: ForecastingModel instance
        background_data: numpy array of shape (N, input_size). 
                         Used to create a background distribution for SHAP.
        n_kmeans: number of centroids to summarize background data (for speed).
        """
        self.model = model
        self.background_summary = shap.kmeans(background_data, n_kmeans)
        # Define the prediction function wrapper
        # SHAP passes a numpy array of shape (n_samples, n_features)
        self.explainer = shap.KernelExplainer(self.batched_predict, self.background_summary)

    def batched_predict(self, data):
        """
        Wrapper to chunk data before sending to model to avoid NeuralForecast batching errors.
        """
        n_samples = len(data)
        batch_size = 32 # Safe batch size that NeuralForecast handles well
        results = []
        
        for i in range(0, n_samples, batch_size):
            chunk = data[i:i+batch_size]
            # Call the model's batch predictor on this small chunk
            chunk_pred = self.model.predict_batch(chunk)
            results.append(chunk_pred)
            
        return np.concatenate(results)

    def explain(self, query_ts, plotting=True, save_path=None):
        """
        Explain a single time series window.
        query_ts: shape (input_size,)
        """
        query_reshaped = query_ts.reshape(1, -1)
        shap_values = self.explainer.shap_values(query_reshaped, nsamples=100)
        
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values
            
        expected_value = self.explainer.expected_value
        
        if plotting:
            self.plot_force(expected_value, sv[0], query_ts, save_path)
            
        return sv, expected_value

    def plot_force(self, expected_value, shap_values, feature_values, save_path=None):
        """
        Custom plot for Time Series SHAP
        """
        plt.figure(figsize=(12, 6))
        
        norm_shap = (shap_values - np.min(shap_values)) / (np.max(shap_values) - np.min(shap_values) + 1e-9)
        
        x = np.arange(len(feature_values))
        
        plt.plot(x, feature_values, 'k-', alpha=0.5, label='Time Series')
        
        # Scatter points colored by impact
        # Blue = Negative Impact (Pushes prediction DOWN)
        plt.scatter(x, feature_values, c=shap_values, cmap='coolwarm', s=100, zorder=5, edgecolor='k')
        
        plt.colorbar(label='SHAP Value (Impact on Prediction)')
        plt.title(f'SHAP Feature Importance (Base Value: {expected_value:.4f})')
        plt.xlabel('Time Steps (t-30 to t-1)')
        plt.ylabel('Log Return')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"SHAP plot saved to {save_path}")
        else:
            plt.show()
