import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

time_steps = np.linspace(0, 100, 500)
data = np.sin(time_steps) + np.random.normal(0, 0.1, 500)

# 2️⃣ Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))


def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)


sequence_length = 50
X, y = create_sequences(data_scaled, sequence_length)

X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(sequence_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(X, y, epochs=30, batch_size=16, validation_split=0.1)

last_sequence = data_scaled[-sequence_length:]
predictions = []

for _ in range(100):
    last_sequence_reshaped = last_sequence.reshape((1, sequence_length, 1))
    next_value = model.predict(last_sequence_reshaped)
    predictions.append(next_value[0, 0])
    last_sequence = np.append(last_sequence[1:], next_value, axis=0)

predictions_rescaled = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(data, label='Original Data')
plt.plot(range(len(data), len(data) + len(predictions_rescaled)),
         predictions_rescaled, label='Forecast')
plt.legend()
plt.title("Time Series Forecasting with LSTM")
plt.show()
