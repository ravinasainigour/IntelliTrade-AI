import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def run_arima(df):

    # ==========================
    # Prepare Data
    # ==========================
    data = df['Close'].dropna()

    train_size = int(len(data) * 0.80)

    train = data[:train_size]
    test = data[train_size:]

    # ==========================
    # Train Model
    # ==========================
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()

    # ==========================
    # Test Prediction
    # ==========================
    predictions = model_fit.forecast(steps=len(test))

    # ==========================
    # Evaluation Metrics
    # ==========================
    rmse = np.sqrt(mean_squared_error(test, predictions))

    mae = mean_absolute_error(test, predictions)

    r2 = r2_score(test, predictions)

    # ==========================
    # Future Forecast (30 Days)
    # ==========================
    final_model = ARIMA(data, order=(5,1,0))
    final_fit = final_model.fit()

    future_forecast = final_fit.forecast(steps=30)

    # ==========================
    # Return Everything
    # ==========================
    return {
        "actual": np.array(test),
        "predicted": np.array(predictions),
        "future": np.array(future_forecast),

        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }