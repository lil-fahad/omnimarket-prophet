
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate, LayerNormalization, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_lstm_advanced(input_shape, output_dim=1, units=128, dropout=0.3):
    inp = Input(shape=input_shape)
    
    x = LSTM(units, return_sequences=True)(inp)
    x = Dropout(dropout)(x)
    x = LSTM(units, return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout)(x)

    # Attention layer
    attention = tf.keras.layers.Attention()([x, x])
    x = Concatenate()([x, attention])

    x = LSTM(units, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    out = Dense(output_dim)(x)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005), metrics=['mae'])
    return model
