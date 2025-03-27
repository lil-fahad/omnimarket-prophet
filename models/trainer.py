
import os
from tensorflow.keras.callbacks import ModelCheckpoint

def train_and_save(model, X_train, y_train, model_path, epochs=20, batch_size=32):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", mode="min")
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    return history
