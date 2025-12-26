import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.model import ForecastingModel
from src.mascots import MASCOTS

def load_data(filepath, ticker):
    # Load Google Finance data
    # Format usually: Date,Open,High,Low,Close,Volume
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Prepare for NeuralForecast
    # unique_id, ds, y
    nf_df = pd.DataFrame({
        'unique_id': ticker,
        'ds': df['Date'],
        'y': df['Close']
    })
    return nf_df

def main():
    ticker = "AAPL"
    data_path = f"data/NASDAQ_{ticker}_30Y.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please run download_data.py first.")
        return

    print(f"Loading data for {ticker}...")
    df = load_data(data_path, ticker)
    
    # Split train/test
    train_size = len(df) - 30
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:] # checking on holdout
    
    # Initialize and Train Model
    model = ForecastingModel(horizon=5, input_size=30)
    model.train(train_df)
    
    # Pick a query instance (the last window of training data to predict the first test point)
    query_ts = train_df['y'].values[-30:] # Last 30 days
    current_pred = model.predict_from_array(query_ts)
    print(f"Current Prediction (Next day price): {current_pred:.2f}")
    
    # Define Counterfactual Goal
    # e.g. We want the price to be 5% higher
    target_price = current_pred * 1.05
    print(f"Goal: Prediction > {target_price:.2f}")
    
    def condition(pred):
        return pred > target_price

    # Run MASCOTS
    mascots = MASCOTS(model)
    cf_ts, cf_pred = mascots.generate_counterfactual(query_ts, condition)
    
    if cf_ts is not None:
        print(f"Counterfactual found! New Prediction: {cf_pred:.2f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(30), query_ts, label='Original History', marker='o')
        plt.plot(range(30), cf_ts, label='Counterfactual History', linestyle='--', marker='x')
        plt.axhline(current_pred, color='blue', linestyle=':', label='Original Pred')
        plt.axhline(cf_pred, color='orange', linestyle=':', label='CF Pred')
        plt.title(f"MASCOTS Explanation for {ticker}")
        plt.legend()
        plt.savefig("mascots_explanation.png")
        print("Saved plot to mascots_explanation.png")
    else:
        print("No counterfactual found within iterations.")

if __name__ == "__main__":
    main()
