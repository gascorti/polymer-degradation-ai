"""
Modelo Random Forest para clasificar la tasa de degradación (baja/media/alta).
HU3 del Product Backlog.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing.features import build_preprocessor


def build_random_forest_pipeline(n_estimators=300, max_depth=12, random_state=42) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])
