"""
Estandarización de variables experimentales (HU2, Criterio de aceptación HU2).

Unifica unidades y formato, y realiza limpieza básica (valores nulos,
outliers evidentes, tipos de dato) sobre el dataset en formato long.
"""

import numpy as np
import pandas as pd

from src.data.schema import LONG_FORMAT_COLUMNS, validate_long_format


def load_raw_curves(paths: list) -> pd.DataFrame:
    """Carga y concatena uno o más CSV en formato long, provenientes de distintas fuentes."""
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def standardize_units(df: pd.DataFrame, numeric_cols: list = None) -> pd.DataFrame:
    """
    Normaliza tipos y rangos:
    - Fuerza tipos numéricos en columnas numéricas.
    - Recorta mass_loss_pct al rango [0, 100].
    - Normaliza strings de categóricas (mayúsculas para polímero, capitalizado para medio).

    `numeric_cols` permite adaptar la lista de columnas a coercionar al esquema
    real (ver src/data/schema.py); por defecto usa el esquema sintético.
    """
    df = df.copy()

    if numeric_cols is None:
        numeric_cols = [
            "temperature_C", "pH", "initial_mw_kDa", "crystallinity_pct",
            "surface_area_mm2", "time_days", "mass_loss_pct",
        ]
    else:
        numeric_cols = numeric_cols + ["time_days", "mass_loss_pct"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["polymer_type"] = df["polymer_type"].astype(str).str.strip().str.upper()
    df["medium"] = df["medium"].astype(str).str.strip().str.capitalize()

    df["mass_loss_pct"] = df["mass_loss_pct"].clip(lower=0, upper=100)
    df = df[df["time_days"] >= 0]

    return df


def handle_missing_and_outliers(df: pd.DataFrame, z_thresh: float = 4.0, numeric_cols: list = None) -> pd.DataFrame:
    """
    Elimina filas con nulos en columnas críticas, imputa nulos en variables
    numéricas por mediana agrupada por tipo de polímero, y filtra outliers
    groseros mediante z-score por curva.

    `numeric_cols` permite adaptar qué columnas imputar al esquema real (ver
    src/data/schema.py); por defecto usa el esquema sintético.
    """
    df = df.copy()
    critical = ["curve_id", "polymer_type", "time_days", "mass_loss_pct"]
    df = df.dropna(subset=critical)

    if numeric_cols is None:
        numeric_cols = ["temperature_C", "pH", "initial_mw_kDa", "crystallinity_pct", "surface_area_mm2"]

    for col in numeric_cols:
        df[col] = df.groupby("polymer_type")[col].transform(lambda s: s.fillna(s.median()))

    grp = df.groupby("curve_id")["mass_loss_pct"]
    z_score = (df["mass_loss_pct"] - grp.transform("mean")) / (grp.transform("std") + 1e-9)
    df = df[z_score.abs() < z_thresh]

    return df.reset_index(drop=True)


def run_standardization_pipeline(input_paths: list, output_path: str) -> pd.DataFrame:
    """Pipeline completo: carga -> validación de esquema -> estandarización -> limpieza -> guardado."""
    df = load_raw_curves(input_paths)

    result = validate_long_format(df)
    if not result.is_valid:
        raise ValueError(f"Dataset no cumple el esquema esperado. Faltan columnas: {result.missing_columns}")
    for w in result.warnings:
        print(f"[WARNING] {w}")

    df = standardize_units(df)
    df = handle_missing_and_outliers(df)

    df.to_csv(output_path, index=False)
    print(f"Dataset estandarizado guardado en {output_path} ({len(df)} filas, {df['curve_id'].nunique()} curvas)")
    return df


if __name__ == "__main__":
    run_standardization_pipeline(
        input_paths=["data/raw/curvas_sinteticas.csv"],
        output_path="data/processed/curvas_estandarizadas.csv",
    )
