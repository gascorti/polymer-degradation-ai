"""
Genera un informe de avance del proyecto en formato Word (.docx), para
presentar a la dirección de tesis. Toma los resultados ya generados por
`main.py` (outputs/reports/comparacion_modelos.csv y outputs/figures/*.png).

Uso:
    python generar_informe.py
"""

import os
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"
OUTPUT_PATH = os.path.join(REPORTS_DIR, "informe_avance_tesis.docx")


def add_title_page(doc):
    doc.add_paragraph()
    title = doc.add_heading(
        "Aplicación de IA para la predicción de tasas de degradación\n"
        "de polímeros biocompatibles",
        level=0,
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph("Informe de avance")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(16)
    sub.runs[0].font.italic = True

    doc.add_paragraph()
    for line in [
        "Trabajo Final — Carrera de Especialización en Inteligencia Artificial (FIUBA)",
        "Autor: Lic. Gastón Corti",
        "Director: Esp. Ing. Ariadna Garmendia",
    ]:
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()


def add_metrics_table(doc, comparison_df):
    table = doc.add_table(rows=1, cols=len(comparison_df.columns) + 1)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text = "Modelo"
    for i, col in enumerate(comparison_df.columns):
        hdr[i + 1].text = col

    for model_name, row in comparison_df.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(model_name)
        for i, col in enumerate(comparison_df.columns):
            val = row[col]
            cells[i + 1].text = "" if pd.isna(val) else f"{val:.4f}"


def add_classification_report_table(doc, report_df):
    cols = ["precision", "recall", "f1-score", "support"]
    table = doc.add_table(rows=1, cols=len(cols) + 1)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text = "Clase"
    for i, col in enumerate(cols):
        hdr[i + 1].text = col

    for class_name, row in report_df.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(class_name)
        for i, col in enumerate(cols):
            val = row[col]
            cells[i + 1].text = f"{int(val)}" if col == "support" else f"{val:.3f}"


def add_figure(doc, filename, caption, width_cm=15):
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        doc.add_paragraph(f"[Figura no encontrada: {filename}]")
        return
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].font.italic = True
    cap.runs[0].font.size = Pt(10)


def build_report():
    comparison_df = pd.read_csv(os.path.join(REPORTS_DIR, "comparacion_modelos.csv"), index_col=0)
    df_curves = pd.read_csv("data/processed/dataset_estandarizado.csv")
    class_counts = df_curves["degradation_class"].value_counts()
    rf_report = pd.read_csv(os.path.join(REPORTS_DIR, "reporte_clasificacion_random_forest.csv"), index_col=0)
    xgb_report = pd.read_csv(os.path.join(REPORTS_DIR, "reporte_clasificacion_xgboost.csv"), index_col=0)

    doc = Document()
    add_title_page(doc)

    doc.add_heading("1. Objetivo del proyecto", level=1)
    doc.add_paragraph(
        "Implementar un pipeline de IA que, a partir de curvas de degradación "
        "(pérdida de masa vs. tiempo) de polímeros biocompatibles en medios "
        "fisiológicos, estime la constante cinética k y clasifique la tasa de "
        "degradación (baja/media/alta), comparando modelos supervisados "
        "(Random Forest, XGBoost) para la clasificación y un modelo de deep "
        "learning (LSTM) para la regresión de k."
    )

    doc.add_heading("2. Estado actual", level=1)
    doc.add_paragraph(
        "El pipeline corre de punta a punta (CRISP-DM completo) con datos "
        "sintéticos generados mediante una cinética de pseudo-primer orden "
        "fisicoquímicamente plausible, mientras avanza en paralelo la extracción "
        "real de curvas desde publicaciones científicas (HU1, Épica 1, pendiente). "
        "Esto permitió desarrollar y validar el resto del pipeline (preprocesamiento, "
        "modelado, evaluación, visualización, tracking de experimentos) sin depender "
        "de tener primero el dataset bibliográfico completo. Cuando la extracción "
        "esté lista, se reemplaza el CSV sintético por el real (mismo esquema) y se "
        "vuelve a correr el pipeline sin cambios de código."
    )

    doc.add_heading("3. Metodología (CRISP-DM)", level=1)
    doc.add_paragraph(
        "El pipeline (main.py) ejecuta las siguientes fases en orden:", style=None
    )
    for item in [
        "Generación / carga de datos (sintéticos, mientras se completa HU1 con datos reales).",
        "Estandarización y limpieza de unidades (HU2).",
        "Ajuste cinético de pseudo-primer orden por curva (mass_loss_pct(t) = 100·(1−exp(−k·t))) "
        "y categorización en clases baja/media/alta según el k ajustado.",
        "Visualizaciones exploratorias: distribución de clases, matriz de correlaciones, "
        "muestra de curvas.",
        "Entrenamiento y comparación de modelos, con tracking de parámetros/métricas/artefactos "
        "en MLflow (HU3, HU4, HU5).",
        "Visualización comparativa de las curvas reconstruidas/clasificadas por cada modelo.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("4. Datos utilizados", level=1)
    doc.add_paragraph(
        f"Dataset sintético actual: {len(df_curves)} curvas, con la siguiente distribución "
        f"de clases de tasa de degradación: baja={class_counts.get('baja', 0)}, "
        f"media={class_counts.get('media', 0)}, alta={class_counts.get('alta', 0)}."
    )
    doc.add_paragraph(
        "Cada curva incluye: tipo de polímero (PLA, PCL, PGA, PLGA, etc.), medio de ensayo "
        "(SBF, PBS, etc.), temperatura, pH, peso molecular inicial, cristalinidad, área "
        "superficial, y la serie temporal de pérdida de masa. Este esquema es el mismo que "
        "se usará al incorporar las curvas reales extraídas de publicaciones (HU1)."
    )

    doc.add_heading("5. Modelos implementados", level=1)
    doc.add_paragraph(
        "Clasificación (baja/media/alta), comparando dos modelos supervisados:",
    )
    for item in [
        "Random Forest (scikit-learn).",
        "XGBoost.",
    ]:
        doc.add_paragraph(item, style="List Bullet")
    doc.add_paragraph(
        "Regresión de la constante cinética k, con una red LSTM (PyTorch) que toma la curva "
        "interpolada de pérdida de masa como secuencia de entrada."
    )
    doc.add_paragraph(
        "Todos los entrenamientos quedan registrados en MLflow (parámetros, métricas y el "
        "modelo serializado como artefacto), lo que permite comparar corridas entre sí desde "
        "la interfaz web de MLflow."
    )

    doc.add_heading("6. Resultados", level=1)
    doc.add_paragraph(
        "Cuadro comparativo de la corrida más reciente (80 curvas sintéticas, semilla 42):"
    )
    add_metrics_table(doc, comparison_df)
    doc.add_paragraph()
    doc.add_paragraph(
        "Random Forest y XGBoost clasifican con accuracy alto (1.00 y 0.94 respectivamente) "
        "sobre el conjunto de test. El LSTM predice k con un error absoluto medio (MAE) de "
        "~0.006 día⁻¹ y R²≈0.64, con mejor ajuste en curvas de degradación rápida que en las "
        "de degradación lenta."
    )

    doc.add_heading("6.1. Distribución de clases y correlaciones", level=2)
    add_figure(doc, "distribucion_clases.png", "Figura 1. Distribución de clases de tasa de degradación.", width_cm=10)
    add_figure(doc, "matriz_correlaciones.png", "Figura 2. Matriz de correlaciones entre variables numéricas.", width_cm=12)

    doc.add_heading("6.2. Curvas de degradación", level=2)
    add_figure(doc, "curvas_degradacion_muestra.png", "Figura 3. Muestra de curvas de degradación del dataset.", width_cm=14)

    doc.add_heading("6.3. Comparación de modelos sobre curvas de test", level=2)
    doc.add_paragraph(
        "Para una muestra de curvas del conjunto de test: a la izquierda, la curva real "
        "(línea con marcadores) contra la reconstruida con el k predicho por el LSTM (punteada); "
        "al centro y a la derecha, la curva real coloreada según la clase que le asignó cada "
        "clasificador (línea sólida = acierto, punteada = error)."
    )
    add_figure(doc, "comparacion_curvas_modelos.png", "Figura 4. Comparación de predicciones de los 3 modelos sobre curvas de test.", width_cm=17)

    doc.add_heading("6.4. Evaluación detallada de clasificación (RF vs XGBoost)", level=2)
    doc.add_paragraph(
        "Además de accuracy y f1_macro (Sección 6), se detalla la matriz de confusión, las "
        "curvas ROC one-vs-rest con AUC por clase, y el precision/recall/f1-score desglosado "
        "por clase, para ambos clasificadores sobre el mismo conjunto de test."
    )

    doc.add_heading("Random Forest", level=3)
    add_classification_report_table(doc, rf_report)
    doc.add_paragraph()
    add_figure(doc, "matriz_confusion_random_forest.png", "Figura 5. Matriz de confusión — Random Forest.", width_cm=9)
    add_figure(doc, "roc_random_forest.png", "Figura 6. Curvas ROC one-vs-rest — Random Forest.", width_cm=9)

    doc.add_heading("XGBoost", level=3)
    add_classification_report_table(doc, xgb_report)
    doc.add_paragraph()
    add_figure(doc, "matriz_confusion_xgboost.png", "Figura 7. Matriz de confusión — XGBoost.", width_cm=9)
    add_figure(doc, "roc_xgboost.png", "Figura 8. Curvas ROC one-vs-rest — XGBoost.", width_cm=9)

    doc.add_heading("8. Próximos pasos", level=1)
    for item in [
        "HU1 (Épica 1): extracción real de curvas desde publicaciones científicas con "
        "WebPlotDigitizer, y reemplazo del dataset sintético.",
        "Re-evaluar los 3 modelos sobre el dataset real y comparar contra el desempeño "
        "obtenido con datos sintéticos.",
        "Ajuste de hiperparámetros (GridSearchCV, ya soportado por el pipeline) sobre el "
        "dataset real.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    doc.save(OUTPUT_PATH)
    print(f"Informe generado: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_report()
