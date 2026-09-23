"""
Carga del dataset real de biodegradación (HU1, Épica 1), extraído de
publicaciones científicas. A diferencia del generador sintético, este
dataset ya viene con un control de calidad curado por columna
(`Nivel_homogeneidad`), que se usa para quedarnos con el subconjunto
comparable a "medios fisiológicos" (Alcance del proyecto, Sección 4).

Convierte el Excel (formato ancho, una fila por punto tiempo/curva con
columnas propias de la extracción bibliográfica) al esquema long usado
por el resto del pipeline (ver REAL_CATEGORICAL_COLUMNS/REAL_NUMERIC_COLUMNS
en src/data/schema.py).
"""

import pandas as pd

# Columnas numéricas de punto (existen ya en el long format) que necesitan
# imputación de nulos reales en standardize.py. No incluye assay_duration_days/
# n_points (se calculan recién en kinetics.py) ni las fracciones de copolímero
# (ya vienen sin nulos, rellenadas con 0 más abajo).
REAL_POINT_NUMERIC_COLUMNS = ["temperature_C", "pH", "initial_mw_kDa", "porosity_pct"]

# Fracciones de copolímero: el nulo es estructural (esa fracción no aplica
# al polímero de esa fila), no un dato faltante -> se imputa con 0.
FRACTION_COLUMNS_MAP = {
    "PLLA_fraction": "plla_fraction",
    "PLGA_fraction": "plga_fraction",
    "PCL_fraction": "pcl_fraction",
    "PEG_fraction": "peg_fraction",
    "Lactide_fraction_in_PLGA": "lactide_fraction_in_plga",
    "Glycolide_fraction_in_PLGA": "glycolide_fraction_in_plga",
}


def load_real_dataset(path: str, homogeneity_levels: list = None) -> pd.DataFrame:
    """
    Lee la hoja 'Dataset' del Excel real, filtra por `Nivel_homogeneidad`
    (por defecto solo 'nucleo': ensayos en medios fisiológicos, sin
    condiciones extremas ni aceleradas) y devuelve un DataFrame en formato
    long con las columnas canónicas del proyecto.
    """
    if homogeneity_levels is None:
        homogeneity_levels = ["nucleo"]

    df = pd.read_excel(path, sheet_name="Dataset")
    df = df[df["Nivel_homogeneidad"].isin(homogeneity_levels)].copy()

    out = pd.DataFrame()
    out["curve_id"] = df["Curve_ID"]
    out["source_doi"] = df["Paper_ID"]
    out["polymer_type"] = df["Polymer_family_norm"]
    out["medium"] = df["Medium_base_norm"]
    out["geometry_type"] = df["Geometry_type_norm"]
    out["fabrication_method"] = df["Fabrication_method_norm"]
    out["temperature_C"] = df["Temperatura_C"]
    out["pH"] = df["pH"]
    out["initial_mw_kDa"] = df["Mw_g_mol"] / 1000.0
    out["porosity_pct"] = df["Porosidad_percent"]
    out["time_days"] = df["Dia"]
    out["mass_loss_pct"] = df["Weight_loss_final"]

    for src_col, dst_col in FRACTION_COLUMNS_MAP.items():
        out[dst_col] = df[src_col].fillna(0.0)

    return out.reset_index(drop=True)
