"""
Preparación de secuencias temporales interpoladas, insumo del modelo LSTM
(HU3 del Product Backlog, componente "Deep Learning (LSTM)" del propósito
del proyecto).

Cada curva (tiempo, % pérdida de masa) tiene puntos irregulares y en
distinta cantidad. Para poder alimentar una LSTM se interpola cada curva
sobre una grilla temporal común de longitud fija (sequence_length),
normalizada entre 0 y 1 respecto de la duración del ensayo.
"""

import numpy as np
import pandas as pd


def interpolate_curve(time_days: np.ndarray, mass_loss_pct: np.ndarray, sequence_length: int) -> np.ndarray:
    """Interpola una curva sobre una grilla temporal normalizada [0, 1] de longitud fija."""
    t_norm = time_days / (time_days.max() + 1e-9)
    grid = np.linspace(0, 1, sequence_length)
    return np.interp(grid, t_norm, mass_loss_pct)


def build_sequence_dataset(df_long: pd.DataFrame, df_curves: pd.DataFrame, sequence_length: int = 10):
    """
    Construye:
      X_seq: array (n_curvas, sequence_length, 1) con la curva interpolada.
      curve_ids: lista de curve_id en el mismo orden que X_seq (para unir con el target).
    """
    sequences = []
    curve_ids = []
    valid_curve_ids = set(df_curves["curve_id"])

    for curve_id, group in df_long.groupby("curve_id"):
        if curve_id not in valid_curve_ids:
            continue
        group = group.sort_values("time_days")
        if len(group) < 3:
            continue
        seq = interpolate_curve(group["time_days"].values, group["mass_loss_pct"].values, sequence_length)
        sequences.append(seq)
        curve_ids.append(curve_id)

    X_seq = np.array(sequences).reshape(-1, sequence_length, 1) / 100.0  # normalizado a [0,1]
    return X_seq, curve_ids
