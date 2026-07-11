"""
Explicabilidad de los modelos de árboles (Random Forest, XGBoost) vía SHAP
(HU6 del Product Backlog). Se usa TreeExplainer, que es exacto (no una
aproximación local como LIME) para modelos basados en árboles.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def compute_shap_explanation(pipeline, X):
    """
    Calcula la explicación SHAP de un pipeline sklearn (preprocesador + modelo
    de árboles) sobre el espacio de features ya transformado (one-hot +
    escalado), que es el que realmente ve el árbol.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap no está instalado. Ejecutá `pip install shap` para usar explicabilidad.")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    feature_names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_transformed)
    explanation.feature_names = feature_names
    return explanation


def plot_shap_global_importance(explanation, model_name: str, top_n: int = 15, output_dir="outputs/figures"):
    """
    Importancia global: promedio de |SHAP| por feature, agregando entre muestras
    (y entre clases, si es multiclase). Se calcula manualmente en vez de usar
    shap.plots.bar directamente, que en explicaciones multiclase de esta versión
    de la librería falla al intentar graficar.
    """
    os.makedirs(output_dir, exist_ok=True)
    values = explanation.values
    mean_abs = np.abs(values).mean(axis=(0, 2)) if values.ndim == 3 else np.abs(values).mean(axis=0)

    order = np.argsort(mean_abs)[::-1][:top_n]
    feature_names = np.array(explanation.feature_names)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(order)), mean_abs[order][::-1], color="teal")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(feature_names[order][::-1], fontsize=8)
    ax.set_xlabel("Promedio de |valor SHAP|")
    ax.set_title(f"Importancia global (SHAP) — {model_name}")
    fig.tight_layout()

    slug = model_name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"shap_global_{slug}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shap_local_explanation(
    explanation, instance_index: int, class_index: int, class_name: str,
    model_name: str, output_dir="outputs/figures",
):
    """Explicación local (waterfall) de una curva de test, para la clase que el modelo le asignó."""
    os.makedirs(output_dir, exist_ok=True)
    exp = explanation[instance_index, :, class_index] if explanation.values.ndim == 3 else explanation[instance_index]
    shap.plots.waterfall(exp, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Explicación local (SHAP) — {model_name}, clase predicha: {class_name}")
    fig.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"shap_local_{slug}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
