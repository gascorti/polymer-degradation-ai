"""
Modelo LSTM (Long Short-Term Memory) para modelar la secuencia temporal de
pérdida de masa y predecir la constante cinética k (regresión).
HU3 del Product Backlog.

Nota: requiere `tensorflow` instalado (ver requirements.txt).
"""

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def build_lstm_model(sequence_length: int, units: int = 32) -> "tf.keras.Model":
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow no está instalado. Ejecutá `pip install tensorflow` para usar este modelo."
        )
    model = models.Sequential([
        layers.Input(shape=(sequence_length, 1)),
        layers.LSTM(units, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),  # predice k (regresión)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm(model, X_train, y_train, X_val, y_val, epochs=60, batch_size=16, random_state=42):
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("tensorflow no está instalado.")
    tf.random.set_seed(random_state)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    return history
