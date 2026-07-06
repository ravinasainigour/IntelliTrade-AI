from prophet import Prophet
import pandas as pd

def run_prophet(df):
    data = df.reset_index()[['Date', 'Close']]
    data.columns = ['ds', 'y']

    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast