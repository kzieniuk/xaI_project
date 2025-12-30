import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer

class ForecastingModel:
    def __init__(self, horizon=1, input_size=30, n_series=1):
        self.horizon = horizon
        self.input_size = input_size
        self.model = NeuralForecast(
            models=[iTransformer(h=horizon, input_size=input_size, n_series=n_series, max_steps=1500, accelerator='auto')],
            freq='D'
        )
        self.is_fitted = False

    def train(self, df):
        """
        Expects a DataFrame with columns ['unique_id', 'ds', 'y']
        """
        print("Training NeuralForecast model...")
        self.model.fit(df=df)
        self.is_fitted = True
        print("Model trained.")

    def predict(self, df_input):
        """
        Predicts the forecast for the given input dataframe.
        df_input must contain the history necessary for prediction.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        forecasts = self.model.predict(df=df_input)
        return forecasts

    def predict_from_array(self, values):
        """
        Predict from a numpy array of shape (input_size,).
        Used for counterfactual search.
        """
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(values), freq='D')
        df = pd.DataFrame({
            'unique_id': ['dummy'] * len(values),
            'ds': dates,
            'y': values
        })
        
        forecasts = self.model.predict(df=df)
        return forecasts['iTransformer'].values[-1]

    def predict_batch(self, batch_values):
        """
        Predict from a batch of numpy arrays.
        Shape: (n_samples, input_size)
        Returns: (n_samples, )
        """
        n_samples, input_size = batch_values.shape
        batch_size = 32 # Process in safer chunks manually
        
        all_forecasts = []
        
        for i in range(0, n_samples, batch_size):
            batch_slice = batch_values[i:i+batch_size]
            current_batch_size = len(batch_slice)
            
            # Vectorized dataframe creation for this chunk
            unique_ids = np.repeat([f"s_{j}" for j in range(current_batch_size)], input_size)
            
            dates_single = pd.date_range(start='2000-01-01', periods=input_size, freq='D')
            dates = np.tile(dates_single, current_batch_size)
            
            y_values = batch_slice.flatten()
            
            df = pd.DataFrame({
                'unique_id': unique_ids,
                'ds': dates,
                'y': y_values
            })
            
            # Predict on this chunk
            # Note: We don't pass batch_size here, we let it process the whole small DF
            # If the chunk is small enough, it won't trigger the internal batching err
            chunk_forecasts = self.model.predict(df=df)
            
            # Extract values in order
            # NeuralForecast usually preserves order of unique_ids passed in if distinct.
            # But to be safe, we sort by our dummy ID
            chunk_forecasts['id_num'] = chunk_forecasts['unique_id'].apply(lambda x: int(x.split('_')[1]))
            chunk_forecasts = chunk_forecasts.sort_values('id_num')
            
            all_forecasts.append(chunk_forecasts['iTransformer'].values)
            
        return np.concatenate(all_forecasts)

    def cross_validation(self, df, n_windows=1, step_size=1):
        """
        Perform cross validation (rolling window evaluation).
        """
        return self.model.cross_validation(df=df, n_windows=n_windows, step_size=step_size)
