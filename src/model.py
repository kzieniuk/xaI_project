import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import matplotlib.pyplot as plt

class ForecastingModel:
    def __init__(self, horizon=5, input_size=30):
        self.horizon = horizon
        self.input_size = input_size
        self.model = NeuralForecast(
            models=[NHITS(h=horizon, input_size=input_size, max_steps=100)],
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
        
        # NeuralForecast predict uses the last available window in df
        forecasts = self.model.predict(df=df_input)
        return forecasts

    def predict_from_array(self, values):
        """
        Predict from a numpy array of shape (input_size,).
        Used for counterfactual search.
        """
        # Create a dummy dataframe for prediction
        # We assume the dates end at today and go back
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(values), freq='D')
        df = pd.DataFrame({
            'unique_id': ['dummy'] * len(values),
            'ds': dates,
            'y': values
        })
        
        forecasts = self.model.predict(df=df)
        # Return the forecast value for the last step (or average)
        # N-HITS returns 'NHITS' column
        return forecasts['NHITS'].values[-1]
