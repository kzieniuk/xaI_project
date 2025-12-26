import os
import requests
import time

DATA_DIR = "data"
BASE_URL = "https://raw.githubusercontent.com/fulifeng/Temporal_Relational_Stock_Ranking/master/data/google_finance/"

FILES_TO_DOWNLOAD = [
    "NASDAQ_AAPL_30Y.csv",
    "NASDAQ_GOOG_30Y.csv",
    "NASDAQ_MSFT_30Y.csv",
    "NASDAQ_AMZN_30Y.csv",
    "NASDAQ_AAL_30Y.csv"
]

def download_file(filename):
    url = BASE_URL + filename
    path = os.path.join(DATA_DIR, filename)
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {path}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for filename in FILES_TO_DOWNLOAD:
        download_file(filename)
        time.sleep(0.5)
