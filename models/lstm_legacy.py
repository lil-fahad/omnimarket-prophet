from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

def train_lstm(X_train, y_train, X_test):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    preds = model.predict(X_test)
    return preds