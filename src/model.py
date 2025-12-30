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
        unique_ids = np.repeat([f"s_{i}" for i in range(n_samples)], input_size)
        
        dates_single = pd.date_range(start='2000-01-01', periods=input_size, freq='D')
        dates = np.tile(dates_single, n_samples)
        
        y_values = batch_values.flatten()
        
        df = pd.DataFrame({
            'unique_id': unique_ids,
            'ds': dates,
            'y': y_values
        })
        
        forecasts = self.model.predict(df=df)
        
        forecasts['id_num'] = forecasts['unique_id'].apply(lambda x: int(x.split('_')[1]))
        forecasts = forecasts.sort_values('id_num')
        
        return forecasts['iTransformer'].values

    def cross_validation(self, df, n_windows=1, step_size=1):
        """
        Perform cross validation (rolling window evaluation).
        """
        return self.model.cross_validation(df=df, n_windows=n_windows, step_size=step_size)
