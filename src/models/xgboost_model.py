"""
Modelo XGBoost para clasificar la tasa de degradación (baja/media/alta).
HU3 del Product Backlog.

Nota: requiere el paquete `xgboost` instalado (ver requirements.txt).
"""

from sklearn.pipeline import Pipeline

from src.preprocessing.features import build_preprocessor

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def build_xgboost_pipeline(
    n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42,
    categorical_cols=None, numeric_cols=None,
) -> Pipeline:
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost no está instalado. Ejecutá `pip install xgboost` para usar este modelo."
        )
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        objective="multi:softprob",
        eval_metric="mlogloss",
    )
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(categorical_cols, numeric_cols)),
        ("model", model),
    ])
