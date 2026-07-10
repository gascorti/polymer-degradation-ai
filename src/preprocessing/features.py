"""
Feature engineering final y división en conjuntos de entrenamiento,
validación y prueba (Sección 8.3 y HU3 del Product Backlog).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.data.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Preprocesador sklearn: one-hot para categóricas, escalado estándar para numéricas."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )


def split_dataset(df: pd.DataFrame, target_col: str, test_size: float, val_size: float, random_state: int):
    """
    Split estratificado (si target es categórico) en train / val / test.
    val_size se interpreta como fracción del total (no del remanente).
    """
    strat = df[target_col] if df[target_col].dtype.name in ("category", "object") else None

    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )

    strat2 = df_train_val[target_col] if strat is not None else None
    relative_val = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_val, test_size=relative_val, random_state=random_state, stratify=strat2
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def get_X_y(df: pd.DataFrame, target_col: str):
    feature_cols = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    X = df[feature_cols]
    y = df[target_col]
    return X, y
