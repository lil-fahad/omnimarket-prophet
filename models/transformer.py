
import tensorflow as tf
from tensorflow.keras import layers, models

def build_transformer_model(input_shape, output_dim=1):
    inputs = layers.Input(shape=input_shape)
    x = layers.LayerNormalization()(inputs)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Dropout(0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
