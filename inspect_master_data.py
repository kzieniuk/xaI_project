import sys
import os
import pickle
import numpy as np

# Add external/MASTER to path
sys.path.append(os.path.abspath("external/MASTER"))

try:
    predict_data_dir = r"external\MASTER\data\opensource"
    universe = 'csi300'
    with open(f'{predict_data_dir}\{universe}_dl_test.pkl', 'rb') as f:
        dl_test = pickle.load(f)
    
    print(f"Data Loaded: {type(dl_test)}")
    
    # Inspect first batch
    # The loader is custom, we need to manually access data
    # base_model DailyBatchSamplerRandom logic:
    # yields indices.
    # dl_test likely has .values or behaves like a dataset
    
    if hasattr(dl_test, 'data'):
        data = dl_test.data
        print(f"Data shape: {data.shape}")
        print(f"Sample 0: {data[0]}")
    else:
        # Try iterating if it's a dataset
        print(f"Length: {len(dl_test)}")
        print(f"Sample 0 shape: {dl_test[0].shape}")
        print(f"Sample 0 content: {dl_test[0]}")

except Exception as e:
    print(f"Error: {e}")
