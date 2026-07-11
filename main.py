"""
Orquestador end-to-end del pipeline, siguiendo las fases de CRISP-DM
descriptas en la Sección 8 del plan de proyecto.

Fases ejecutadas:
  1. (Generación de datos sintéticos, mientras se completa HU1 real)
  2. Estandarización y limpieza (HU2)
  3. Ajuste cinético + categorización (Épica 2/3)
  4. Split train/val/test
  5. Entrenamiento de modelos + comparación (HU3, HU4, HU5)
  6. Visualizaciones exploratorias y explicativas (HU6)

Uso:
    python main.py --n_curves 80 --seed 42
"""

import argparse
import yaml
import pandas as pd

from src.data.generate_synthetic import generate_dataset
from src.preprocessing.standardize import standardize_units, handle_missing_and_outliers
from src.preprocessing.kinetics import build_curve_level_dataset, categorize_degradation_rate
from src.visualization.plots import (
    plot_class_distribution, plot_correlation_heatmap, plot_degradation_curves,
    plot_model_curves_comparison, plot_confusion_matrix, plot_roc_curves,
)
from src.visualization.explainability import plot_shap_global_importance, plot_shap_local_explanation
from src.models.train import run_full_training


def main(n_curves: int, seed: int, config_path: str, use_synthetic: bool):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("=== 1. Datos ===")
    if use_synthetic:
        df_long, _ = generate_dataset(n_curves=n_curves, seed=seed)
        df_long.to_csv("data/raw/curvas_sinteticas.csv", index=False)
        print(f"Generadas {n_curves} curvas sintéticas -> data/raw/curvas_sinteticas.csv")
    else:
        df_long = pd.read_csv("data/raw/curvas_sinteticas.csv")

    print("\n=== 2. Estandarización (HU2) ===")
    df_long = standardize_units(df_long)
    df_long = handle_missing_and_outliers(df_long)
    df_long.to_csv("data/processed/curvas_estandarizadas.csv", index=False)
    print(f"{len(df_long)} puntos válidos, {df_long['curve_id'].nunique()} curvas.")

    print("\n=== 3. Ajuste cinético y categorización ===")
    df_curves = build_curve_level_dataset(df_long)
    df_curves = categorize_degradation_rate(
        df_curves,
        bins=cfg["data"]["degradation_class_bins"],
        labels=cfg["data"]["degradation_class_labels"],
    )
    df_curves.to_csv(cfg["paths"]["processed_data"], index=False)
    print(f"{len(df_curves)} curvas con k ajustado.")
    print(df_curves["degradation_class"].value_counts())

    print("\n=== 4. Visualizaciones exploratorias (HU6) ===")
    numeric_cols = ["temperature_C", "pH", "initial_mw_kDa", "crystallinity_pct",
                     "surface_area_mm2", "k_day_inv"]
    p1 = plot_class_distribution(df_curves, "degradation_class")
    p2 = plot_correlation_heatmap(df_curves, numeric_cols)
    p3 = plot_degradation_curves(df_long)
    print(f"Figuras guardadas:\n - {p1}\n - {p2}\n - {p3}")

    print("\n=== 5. Entrenamiento y comparación de modelos (HU3, HU4, HU5) ===")
    results, extras = run_full_training(config_path)

    print("\n=== 6. Curvas reconstruidas por modelo ===")
    p4 = plot_model_curves_comparison(
        extras["df_long"], extras["df_test"],
        rf_pred=extras["rf_pred"], xgb_pred=extras["xgb_pred"], lstm_pred_k=extras["lstm_pred_k"],
    )
    print(f"Figura guardada: {p4}")

    print("\n=== 7. Evaluación detallada de clasificación (RF vs XGBoost) ===")
    labels = extras["labels"]
    for name, ev in [("Random Forest", extras["rf_eval"]), ("XGBoost", extras["xgb_eval"])]:
        if ev is None:
            continue
        p_cm = plot_confusion_matrix(ev["y_true"], ev["y_pred"], labels, name)
        p_roc = plot_roc_curves(ev["y_true"], ev["y_proba"], ev["classes"], name)
        print(f"{name}: {p_cm}, {p_roc}")
        print(ev["report"])

    print("\n=== 8. Explicabilidad (SHAP) ===")
    for name, ev in [("Random Forest", extras["rf_eval"]), ("XGBoost", extras["xgb_eval"])]:
        if ev is None or ev["shap"] is None:
            continue
        p_global = plot_shap_global_importance(ev["shap"], name)
        instance_idx = 0
        predicted_class = ev["y_pred"][instance_idx]
        class_idx = ev["classes"].index(predicted_class)
        p_local = plot_shap_local_explanation(ev["shap"], instance_idx, class_idx, predicted_class, name)
        print(f"{name}: {p_global}, {p_local}")

    print("\nPipeline completo ejecutado con éxito.")
    return df_curves, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline completo del proyecto de degradación de polímeros.")
    parser.add_argument("--n_curves", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--no-synthetic", dest="use_synthetic", action="store_false",
                         help="Usar data/raw/curvas_sinteticas.csv existente en lugar de generar datos nuevos.")
    args = parser.parse_args()
    main(args.n_curves, args.seed, args.config, args.use_synthetic)
