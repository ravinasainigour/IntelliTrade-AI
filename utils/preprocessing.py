import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    df = pd.read_csv(path)

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    return df


def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.fillna(method='ffill')

    return df


def normalize_data(df):
    scaler = MinMaxScaler()

    df_scaled = scaler.fit_transform(df[['Close']])

    return df_scaled, scaler