"""
Modelo Random Forest para clasificar la tasa de degradación (baja/media/alta).
HU3 del Product Backlog.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing.features import build_preprocessor


def build_random_forest_pipeline(
    n_estimators=300, max_depth=12, random_state=42,
    categorical_cols=None, numeric_cols=None,
) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(categorical_cols, numeric_cols)),
        ("model", model),
    ])


def build_random_forest_regressor_pipeline(
    n_estimators=300, max_depth=12, random_state=42,
    categorical_cols=None, numeric_cols=None,
) -> Pipeline:
    """
    Variante de regresión: predice k directamente a partir de las propiedades
    tabulares, sin pasar por la curva. Se usa únicamente para reconstruir una
    curva de referencia en el predictor interactivo de la interfaz Streamlit
    (app.py), para un polímero hipotético del que no existe una curva real.
    No reemplaza ni participa de la comparación de modelos del pipeline
    principal (ver src/models/train.py), que sigue siendo exclusivamente
    Random Forest/XGBoost (clasificación) y LSTM (regresión por secuencia).
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(categorical_cols, numeric_cols)),
        ("model", model),
    ])
