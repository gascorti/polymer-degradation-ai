"""
Visualizaciones exploratorias y explicativas (HU2, HU6 del Product Backlog).
Todas las funciones guardan la figura en `outputs/figures/` y devuelven la ruta.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def _save(fig, name, output_dir="outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(df: pd.DataFrame, target_col: str, output_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=(5, 4))
    order = df[target_col].value_counts().index
    sns.countplot(data=df, x=target_col, hue=target_col, order=order, palette="viridis", legend=False, ax=ax)
    ax.set_title("Distribución de clases de tasa de degradación")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad de curvas")
    return _save(fig, "distribucion_clases.png", output_dir)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list, output_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlaciones")
    return _save(fig, "matriz_correlaciones.png", output_dir)


def plot_degradation_curves(df_long: pd.DataFrame, n_curves: int = 12, output_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sample_ids = df_long["curve_id"].drop_duplicates().sample(
        min(n_curves, df_long["curve_id"].nunique()), random_state=42
    )
    for cid in sample_ids:
        g = df_long[df_long["curve_id"] == cid].sort_values("time_days")
        ax.plot(g["time_days"], g["mass_loss_pct"], marker="o", markersize=3, alpha=0.7, label=cid)
    ax.set_xlabel("Tiempo (días)")
    ax.set_ylabel("Pérdida de masa (%)")
    ax.set_title("Curvas de degradación (muestra)")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    return _save(fig, "curvas_degradacion_muestra.png", output_dir)


def plot_feature_importance(model, feature_names, top_n=15, output_dir="outputs/figures"):
    """Espera un modelo sklearn con atributo `feature_importances_` (Random Forest / XGBoost)."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(order)), importances[order][::-1], color="teal")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(np.array(feature_names)[order][::-1], fontsize=8)
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de variables")
    return _save(fig, "importancia_variables.png", output_dir)


def plot_model_comparison(comparison_df: pd.DataFrame, metric_col: str, output_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=(6, 4))
    comparison_df[metric_col].plot(kind="bar", ax=ax, color="slateblue")
    ax.set_ylabel(metric_col)
    ax.set_title(f"Comparación de modelos — {metric_col}")
    plt.xticks(rotation=20)
    return _save(fig, f"comparacion_modelos_{metric_col}.png", output_dir)
