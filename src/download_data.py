import os
import yfinance as yf
import pandas as pd

DATA_DIR = "data"

TICKERS = [
    "AAPL",
    "GOOG",
    "MSFT",
    "AMZN",
    "AAL"
]

def download_ticker(ticker):
    print(f"Downloading {ticker} from Yahoo Finance...")
    try:
        # data = yf.download(ticker, start="2000-01-01") # Caused issues
        
        # Use Ticker object and history() method
        dat = yf.Ticker(ticker)
        df = dat.history(start="2000-01-01")
        
        if df.empty:
            print(f"No data found for {ticker}")
            return

        # yfinance returns MultiIndex columns in new versions mostly, but let's flatten or handle it
        # usually simpler to just save directly
        # Ensure we have a standard format: Date (index), Open, High, Low, Close, Volume
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Save to CSV using the same naming convention as before to be compatible with main.py
        # Convention: NASDAQ_{ticker}_30Y.csv (even though it's not strictly 30Y or NASDAQ specific source now)
        filename = f"NASDAQ_{ticker}_30Y.csv"
        path = os.path.join(DATA_DIR, filename)
        
        df.to_csv(path, index=False)
        print(f"Saved to {path} ({len(df)} rows)")
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print("Starting download...")
    for ticker in TICKERS:
        download_ticker(ticker)
    print("Download complete.")
