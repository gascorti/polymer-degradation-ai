"""
Ajuste de modelos cinéticos sobre cada curva de degradación (Sección 8.4 del plan,
"pseudo-primer orden o ley exponencial") y categorización en clases baja/media/alta
(Épica 2 y 3 del Product Backlog).

Modelo: mass_loss_pct(t) = 100 * (1 - exp(-k * t))
Se ajusta k por mínimos cuadrados no lineales (scipy.optimize.curve_fit) para
cada curve_id, y se calcula el R² del ajuste como indicador de calidad.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _pseudo_first_order(t, k):
    return 100 * (1 - np.exp(-k * t))


def fit_k_for_curve(time_days: np.ndarray, mass_loss_pct: np.ndarray):
    """Ajusta k para una curva individual. Devuelve (k, r2)."""
    try:
        popt, _ = curve_fit(
            _pseudo_first_order, time_days, mass_loss_pct,
            p0=[0.01], bounds=(1e-6, 1.0), maxfev=5000,
        )
        k = float(popt[0])
        pred = _pseudo_first_order(time_days, k)
        ss_res = np.sum((mass_loss_pct - pred) ** 2)
        ss_tot = np.sum((mass_loss_pct - np.mean(mass_loss_pct)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return k, r2
    except RuntimeError:
        return np.nan, np.nan


def build_curve_level_dataset(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el dataset long (punto a punto) en un dataset wide (una fila por curva),
    con las propiedades del material/ensayo y el parámetro cinético k ajustado.
    """
    records = []
    for curve_id, group in df_long.groupby("curve_id"):
        group = group.sort_values("time_days")
        k, r2 = fit_k_for_curve(group["time_days"].values, group["mass_loss_pct"].values)

        records.append({
            "curve_id": curve_id,
            "polymer_type": group["polymer_type"].iloc[0],
            "medium": group["medium"].iloc[0],
            "temperature_C": group["temperature_C"].iloc[0],
            "pH": group["pH"].iloc[0],
            "initial_mw_kDa": group["initial_mw_kDa"].iloc[0],
            "crystallinity_pct": group["crystallinity_pct"].iloc[0],
            "surface_area_mm2": group["surface_area_mm2"].iloc[0],
            "assay_duration_days": group["time_days"].max(),
            "n_points": len(group),
            "k_day_inv": k,
            "kinetic_fit_r2": r2,
        })

    df_curves = pd.DataFrame(records)
    df_curves = df_curves.dropna(subset=["k_day_inv"])
    return df_curves


def categorize_degradation_rate(df_curves: pd.DataFrame, bins: list, labels: list) -> pd.DataFrame:
    """Asigna la clase (baja/media/alta) según el valor de k, usando los bordes de config.yaml."""
    df_curves = df_curves.copy()
    df_curves["degradation_class"] = pd.cut(
        df_curves["k_day_inv"], bins=bins, labels=labels, include_lowest=True
    )
    return df_curves


if __name__ == "__main__":
    df_long = pd.read_csv("data/processed/curvas_estandarizadas.csv")
    df_curves = build_curve_level_dataset(df_long)
    df_curves = categorize_degradation_rate(
        df_curves, bins=[0.0, 0.01, 0.03, 1.0], labels=["baja", "media", "alta"]
    )
    df_curves.to_csv("data/processed/dataset_estandarizado.csv", index=False)
    print(f"Dataset a nivel de curva generado: {len(df_curves)} curvas.")
    print(df_curves["degradation_class"].value_counts())
