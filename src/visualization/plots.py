"""
Visualizaciones exploratorias y explicativas (HU2, HU6 del Product Backlog).
Todas las funciones guardan la figura en `outputs/figures/` y devuelven la ruta.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

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


def _plot_classified_curves(ax, df_long, df_test, pred_series, sample_ids, class_colors):
    """Dibuja la curva real de cada `sample_ids`, coloreada según la clase que le asignó
    el clasificador (sólida si acertó la clase real, punteada si no)."""
    if pred_series is None:
        ax.text(0.5, 0.5, "Modelo no disponible", ha="center", va="center", transform=ax.transAxes)
        return
    df_test_idx = df_test.set_index("curve_id")
    for cid in sample_ids:
        if cid not in pred_series.index:
            continue
        g = df_long[df_long["curve_id"] == cid].sort_values("time_days")
        pred_class = pred_series.loc[cid]
        true_class = df_test_idx.loc[cid, "degradation_class"]
        linestyle = "-" if pred_class == true_class else "--"
        ax.plot(
            g["time_days"], g["mass_loss_pct"], linestyle, marker="o", markersize=3,
            color=class_colors.get(pred_class, "gray"), alpha=0.8,
        )
    handles = [plt.Line2D([0], [0], color=c, label=lbl) for lbl, c in class_colors.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower right", title="Clase predicha")


def plot_model_curves_comparison(
    df_long: pd.DataFrame,
    df_test: pd.DataFrame,
    rf_pred: pd.Series = None,
    xgb_pred: pd.Series = None,
    lstm_pred_k: pd.Series = None,
    n_curves: int = 6,
    output_dir="outputs/figures",
):
    """
    Panel comparativo sobre una muestra de curvas de test:
      - LSTM: curva real (o) vs. curva reconstruida con el k predicho, vía
        mass_loss_pct(t) = 100*(1 - exp(-k*t)) (--).
      - Random Forest / XGBoost: curva real coloreada según la clase predicha
        (línea sólida = acierto, punteada = error de clasificación).
    """
    class_colors = {"baja": "#2ca02c", "media": "#ff7f0e", "alta": "#d62728"}
    sample_ids = df_test["curve_id"].drop_duplicates().sample(
        min(n_curves, df_test["curve_id"].nunique()), random_state=42
    ).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    ax = axes[0]
    for cid in sample_ids:
        if lstm_pred_k is None or cid not in lstm_pred_k.index:
            continue
        g = df_long[df_long["curve_id"] == cid].sort_values("time_days")
        t = g["time_days"].values
        line, = ax.plot(t, g["mass_loss_pct"].values, "o-", markersize=3, alpha=0.6)
        t_smooth = np.linspace(0, t.max(), 50)
        curve_pred = 100 * (1 - np.exp(-lstm_pred_k.loc[cid] * t_smooth))
        ax.plot(t_smooth, curve_pred, "--", alpha=0.8, color=line.get_color())
    ax.set_title("LSTM: real (o-) vs. reconstruida con k predicho (--)")
    ax.set_xlabel("Tiempo (días)")
    ax.set_ylabel("Pérdida de masa (%)")

    ax = axes[1]
    _plot_classified_curves(ax, df_long, df_test, rf_pred, sample_ids, class_colors)
    ax.set_title("Random Forest\n(sólida = acierto, punteada = error)")
    ax.set_xlabel("Tiempo (días)")

    ax = axes[2]
    _plot_classified_curves(ax, df_long, df_test, xgb_pred, sample_ids, class_colors)
    ax.set_title("XGBoost\n(sólida = acierto, punteada = error)")
    ax.set_xlabel("Tiempo (días)")

    fig.tight_layout()
    return _save(fig, "comparacion_curvas_modelos.png", output_dir)


def plot_confusion_matrix(y_true, y_pred, labels: list, model_name: str, output_dir="outputs/figures"):
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión — {model_name}")
    slug = model_name.lower().replace(" ", "_")
    return _save(fig, f"matriz_confusion_{slug}.png", output_dir)


def plot_roc_curves(y_true, y_proba, classes: list, model_name: str, output_dir="outputs/figures"):
    """y_proba: array (n_samples, n_classes) en el mismo orden que `classes`."""
    class_colors = {"baja": "#2ca02c", "media": "#ff7f0e", "alta": "#d62728"}
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=class_colors.get(cls), label=f"{cls} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curvas ROC one-vs-rest — {model_name}")
    ax.legend(fontsize=8, loc="lower right")
    slug = model_name.lower().replace(" ", "_")
    return _save(fig, f"roc_{slug}.png", output_dir)


def plot_model_comparison(comparison_df: pd.DataFrame, metric_col: str, output_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=(6, 4))
    comparison_df[metric_col].plot(kind="bar", ax=ax, color="slateblue")
    ax.set_ylabel(metric_col)
    ax.set_title(f"Comparación de modelos — {metric_col}")
    plt.xticks(rotation=20)
    return _save(fig, f"comparacion_modelos_{metric_col}.png", output_dir)
