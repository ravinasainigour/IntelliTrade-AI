import pandas as pd

def add_indicators(df):
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    df['EMA_20'] = df['Close'].ewm(span=20).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df