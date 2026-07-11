"""
Tests del pipeline. Cubren generación de datos, estandarización, ajuste
cinético, split y entrenamiento de Random Forest (el único modelo que no
depende de librerías externas pesadas como xgboost/torch).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.data.generate_synthetic import generate_dataset
from src.data.schema import validate_long_format
from src.preprocessing.standardize import standardize_units, handle_missing_and_outliers
from src.preprocessing.kinetics import build_curve_level_dataset, categorize_degradation_rate, fit_k_for_curve
from src.preprocessing.features import split_dataset, get_X_y
from src.models.random_forest import build_random_forest_pipeline
from src.evaluation.metrics import classification_metrics


def test_generate_dataset_schema():
    df, true_k = generate_dataset(n_curves=10, seed=1)
    result = validate_long_format(df)
    assert result.is_valid, f"Faltan columnas: {result.missing_columns}"
    assert df["curve_id"].nunique() == 10
    assert len(true_k) == 10


def test_standardize_and_clean():
    df, _ = generate_dataset(n_curves=15, seed=2)
    df_std = standardize_units(df)
    df_clean = handle_missing_and_outliers(df_std)
    assert df_clean["mass_loss_pct"].between(0, 100).all()
    assert df_clean["time_days"].ge(0).all()
    assert df_clean.isna().sum().sum() == 0


def test_kinetic_fit_recovers_k_reasonably():
    """El ajuste cinético debería recuperar aproximadamente el k real usado para simular."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 90, 12))
    t[0] = 0
    k_real = 0.02
    m = 100 * (1 - np.exp(-k_real * t))
    k_fit, r2 = fit_k_for_curve(t, m)
    assert abs(k_fit - k_real) / k_real < 0.05
    assert r2 > 0.95


def test_curve_level_dataset_and_categorization():
    df, _ = generate_dataset(n_curves=30, seed=3)
    df = standardize_units(df)
    df = handle_missing_and_outliers(df)
    df_curves = build_curve_level_dataset(df)
    assert "k_day_inv" in df_curves.columns
    assert len(df_curves) > 0

    df_cat = categorize_degradation_rate(
        df_curves, bins=[0.0, 0.01, 0.03, 1.0], labels=["baja", "media", "alta"]
    )
    assert set(df_cat["degradation_class"].dropna().unique()) <= {"baja", "media", "alta"}
    assert df_cat["degradation_class"].isna().sum() == 0


def test_split_dataset_proportions():
    df, _ = generate_dataset(n_curves=100, seed=4)
    df = standardize_units(df)
    df = handle_missing_and_outliers(df)
    df_curves = build_curve_level_dataset(df)
    df_curves = categorize_degradation_rate(
        df_curves, bins=[0.0, 0.01, 0.03, 1.0], labels=["baja", "media", "alta"]
    )
    df_train, df_val, df_test = split_dataset(
        df_curves, target_col="degradation_class", test_size=0.2, val_size=0.1, random_state=42
    )
    total = len(df_train) + len(df_val) + len(df_test)
    assert total == len(df_curves)
    assert len(df_test) / total == pytest_approx(0.2, abs_tol=0.05)


def pytest_approx(value, abs_tol):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= abs_tol
    return _Approx()


def test_random_forest_end_to_end():
    """Entrena y evalúa un Random Forest end-to-end sobre datos sintéticos."""
    df, _ = generate_dataset(n_curves=120, seed=5)
    df = standardize_units(df)
    df = handle_missing_and_outliers(df)
    df_curves = build_curve_level_dataset(df)
    df_curves = categorize_degradation_rate(
        df_curves, bins=[0.0, 0.01, 0.03, 1.0], labels=["baja", "media", "alta"]
    )

    df_train, df_val, df_test = split_dataset(
        df_curves, target_col="degradation_class", test_size=0.2, val_size=0.1, random_state=42
    )
    X_train, y_train = get_X_y(df_train, "degradation_class")
    X_test, y_test = get_X_y(df_test, "degradation_class")

    pipeline = build_random_forest_pipeline(n_estimators=100, max_depth=8, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = classification_metrics(y_test, y_pred, labels=sorted(df_curves["degradation_class"].unique()))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_lstm_end_to_end():
    """Entrena y evalúa la LSTM (PyTorch) end-to-end. Se salta si torch no está instalado."""
    from src.models.lstm_model import build_lstm_model, train_lstm, TORCH_AVAILABLE
    from src.preprocessing.sequences import build_sequence_dataset
    from src.evaluation.metrics import regression_metrics

    if not TORCH_AVAILABLE:
        print("[SKIP] torch no instalado, se omite test_lstm_end_to_end.")
        return

    df, _ = generate_dataset(n_curves=60, seed=6)
    df = standardize_units(df)
    df = handle_missing_and_outliers(df)
    df_curves = build_curve_level_dataset(df)
    df_curves = categorize_degradation_rate(
        df_curves, bins=[0.0, 0.01, 0.03, 1.0], labels=["baja", "media", "alta"]
    )

    df_train, df_val, df_test = split_dataset(
        df_curves, target_col="degradation_class", test_size=0.2, val_size=0.1, random_state=42
    )

    seq_len = 10
    X_train, ids_train = build_sequence_dataset(df, df_train, seq_len)
    X_val, ids_val = build_sequence_dataset(df, df_val, seq_len)
    X_test, ids_test = build_sequence_dataset(df, df_test, seq_len)

    y_train = df_train.set_index("curve_id").loc[ids_train, "k_day_inv"].values
    y_val = df_val.set_index("curve_id").loc[ids_val, "k_day_inv"].values
    y_test = df_test.set_index("curve_id").loc[ids_test, "k_day_inv"].values

    model = build_lstm_model(sequence_length=seq_len, units=16)
    train_lstm(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=8, random_state=42)
    y_pred = model.predict(X_test, verbose=0).flatten()

    assert y_pred.shape == y_test.shape
    metrics = regression_metrics(y_test, y_pred)
    assert metrics["rmse"] >= 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
