"""
Feature engineering final y división en conjuntos de entrenamiento,
validación y prueba (Sección 8.3 y HU3 del Product Backlog).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.data.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor(categorical_cols: list = None, numeric_cols: list = None) -> ColumnTransformer:
    """
    Preprocesador sklearn: one-hot para categóricas, escalado estándar para numéricas.
    Por defecto usa el esquema sintético (src/data/schema.py); pasar
    REAL_CATEGORICAL_COLUMNS/REAL_NUMERIC_COLUMNS para el dataset real.
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLUMNS
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLUMNS

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
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


def get_X_y(df: pd.DataFrame, target_col: str, categorical_cols: list = None, numeric_cols: list = None):
    feature_cols = (categorical_cols or CATEGORICAL_COLUMNS) + (numeric_cols or NUMERIC_COLUMNS)
    X = df[feature_cols]
    y = df[target_col]
    return X, y
