"""
Cálculo de métricas de clasificación y regresión, y armado del cuadro
comparativo entre modelos (HU5, Criterio de aceptación HU5).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def classification_report_table(y_true, y_pred, labels) -> pd.DataFrame:
    """Precision/recall/f1-score/support por clase, como DataFrame (una fila por clase)."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.loc[labels]


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    results: {"Random Forest": {"accuracy": .., "f1_macro": ..}, "XGBoost": {...}, ...}
    Devuelve un DataFrame ordenado por f1_macro (o rmse si es regresión) descendente/ascendente.
    """
    df = pd.DataFrame(results).T
    if "f1_macro" in df.columns:
        df = df.sort_values("f1_macro", ascending=False)
    elif "rmse" in df.columns:
        df = df.sort_values("rmse", ascending=True)
    return df
