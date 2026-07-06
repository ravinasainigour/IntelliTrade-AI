import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# =========================
# CREATE DATASET
# =========================

def create_dataset(dataset, time_step=60):

    X, y = [], []

    for i in range(len(dataset) - time_step - 1):

        X.append(dataset[i:(i + time_step), 0])

        y.append(dataset[i + time_step, 0])

    return np.array(X), np.array(y)


# =========================
# RUN LSTM
# =========================

def run_lstm(df_scaled, scaler):

    time_step = 60

    X, y = create_dataset(df_scaled, time_step)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    # =========================
    # MODEL
    # =========================

    model = Sequential()

    model.add(LSTM(
        50,
        return_sequences=True,
        input_shape=(time_step, 1)
    ))

    model.add(LSTM(50))

    model.add(Dense(1))

    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )

    model.fit(
        X,
        y,
        epochs=5,
        batch_size=32,
        verbose=0
    )

    # =========================
    # FUTURE PREDICTION
    # =========================

    temp_input = list(df_scaled[-60:].flatten())

    future_output = []

    for i in range(30):

        x_input = np.array(temp_input[-60:])

        x_input = x_input.reshape(1, 60, 1)

        yhat = model.predict(x_input, verbose=0)

        temp_input.append(yhat[0][0])

        future_output.append(yhat[0][0])

    # =========================
    # INVERSE TRANSFORM
    # =========================

    future_output = np.array(future_output)

    future_output = future_output.reshape(-1, 1)

    future_output = scaler.inverse_transform(future_output)

    return future_output.flatten()