"""
Interfaz Streamlit para comparar la degradación de distintos polímeros
biocompatibles (curvas + categoría baja/media/alta) sobre el dataset REAL
(HU1), y predecir la categoría de un polímero hipotético con los modelos
entrenados (Random Forest / XGBoost).

Uso:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.data.load_real_dataset import load_real_dataset, REAL_POINT_NUMERIC_COLUMNS
from src.data.schema import REAL_CATEGORICAL_COLUMNS, REAL_NUMERIC_COLUMNS
from src.preprocessing.standardize import standardize_units, handle_missing_and_outliers
from src.preprocessing.kinetics import build_curve_level_dataset, categorize_degradation_rate
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost_model import build_xgboost_pipeline, XGBOOST_AVAILABLE

CONFIG_PATH = "config/config_real.yaml"
FRACTION_COLUMNS = [
    "plla_fraction", "plga_fraction", "pcl_fraction", "peg_fraction",
    "lactide_fraction_in_plga", "glycolide_fraction_in_plga",
]
CLASS_COLORS = {"baja": "#2ca02c", "media": "#ff7f0e", "alta": "#d62728"}


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data():
    cfg = load_config()
    df_long = load_real_dataset(cfg["data"]["real_dataset_path"], cfg["data"]["homogeneity_levels"])
    df_long = standardize_units(df_long, numeric_cols=REAL_POINT_NUMERIC_COLUMNS)
    df_long = handle_missing_and_outliers(df_long, numeric_cols=REAL_POINT_NUMERIC_COLUMNS)

    metadata_cols = REAL_CATEGORICAL_COLUMNS + [
        c for c in REAL_NUMERIC_COLUMNS if c not in ("assay_duration_days", "n_points")
    ]
    df_curves = build_curve_level_dataset(df_long, metadata_cols=metadata_cols)
    df_curves = categorize_degradation_rate(
        df_curves,
        bins=cfg["data"]["degradation_class_bins"],
        labels=cfg["data"]["degradation_class_labels"],
    )
    return df_long, df_curves


@st.cache_resource
def train_models(df_curves: pd.DataFrame, _cfg: dict):
    feature_cols = REAL_CATEGORICAL_COLUMNS + REAL_NUMERIC_COLUMNS
    X = df_curves[feature_cols]
    y = df_curves["degradation_class"]

    rf_cfg = _cfg["models"]["random_forest"]
    rf_pipeline = build_random_forest_pipeline(
        **rf_cfg, categorical_cols=REAL_CATEGORICAL_COLUMNS, numeric_cols=REAL_NUMERIC_COLUMNS
    )
    rf_pipeline.fit(X, y)
    models = {"Random Forest": {"pipeline": rf_pipeline, "classes": list(rf_pipeline.classes_)}}

    if XGBOOST_AVAILABLE:
        classes = sorted(y.unique())
        idx_to_class = dict(enumerate(classes))
        y_num = y.map({v: k for k, v in idx_to_class.items()})
        xgb_cfg = _cfg["models"]["xgboost"]
        xgb_pipeline = build_xgboost_pipeline(
            **xgb_cfg, categorical_cols=REAL_CATEGORICAL_COLUMNS, numeric_cols=REAL_NUMERIC_COLUMNS
        )
        xgb_pipeline.fit(X, y_num)
        models["XGBoost"] = {"pipeline": xgb_pipeline, "classes": [idx_to_class[i] for i in range(len(classes))]}

    return models


def reconstructed_curve(k, t_max, n=50):
    t = np.linspace(0, t_max, n)
    return t, 100 * (1 - np.exp(-k * t))


st.set_page_config(page_title="Degradación de polímeros biocompatibles", layout="wide")
st.title("Degradación de polímeros biocompatibles")
st.caption("Dataset real (HU1) — subconjunto 'núcleo', comparable a medios fisiológicos.")

cfg = load_config()
df_long, df_curves = load_data()

tab_compare, tab_predict = st.tabs(["Comparar curvas de degradación", "Predictor interactivo"])

# --- Pestaña 1: comparar curvas ---
with tab_compare:
    default_polymers = df_curves["polymer_type"].value_counts().head(4).index.tolist()
    all_polymers = sorted(df_curves["polymer_type"].unique())
    all_media = sorted(df_curves["medium"].unique())

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_polymers = st.multiselect("Tipo de polímero", all_polymers, default=default_polymers)
    with col_f2:
        selected_media = st.multiselect("Medio de ensayo", all_media, default=all_media)

    filtered = df_curves[
        df_curves["polymer_type"].isin(selected_polymers) & df_curves["medium"].isin(selected_media)
    ]

    if filtered.empty:
        st.warning("No hay curvas para los filtros seleccionados.")
    else:
        polymer_list = sorted(filtered["polymer_type"].unique())
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                   "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        color_map = {p: palette[i % len(palette)] for i, p in enumerate(polymer_list)}

        fig = go.Figure()
        seen_legend = set()
        for _, curve in filtered.iterrows():
            cid = curve["curve_id"]
            color = color_map[curve["polymer_type"]]
            points = df_long[df_long["curve_id"] == cid].sort_values("time_days")

            show_legend = curve["polymer_type"] not in seen_legend
            seen_legend.add(curve["polymer_type"])

            fig.add_trace(go.Scatter(
                x=points["time_days"], y=points["mass_loss_pct"], mode="markers",
                marker=dict(color=color, size=5), name=curve["polymer_type"],
                legendgroup=curve["polymer_type"], showlegend=show_legend,
                hovertext=f"{cid} — clase: {curve['degradation_class']}, k={curve['k_day_inv']:.5f}",
                hoverinfo="x+y+text",
            ))
            t_smooth, y_smooth = reconstructed_curve(curve["k_day_inv"], curve["assay_duration_days"])
            fig.add_trace(go.Scatter(
                x=t_smooth, y=y_smooth, mode="lines", line=dict(color=color, width=1, dash="dot"),
                legendgroup=curve["polymer_type"], showlegend=False, hoverinfo="skip",
            ))

        fig.update_layout(
            xaxis_title="Tiempo (días)", yaxis_title="Pérdida de masa (%)",
            legend_title="Tipo de polímero", height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Curvas y categorías")
        st.dataframe(
            filtered[["curve_id", "polymer_type", "medium", "k_day_inv", "degradation_class"]]
            .sort_values(["polymer_type", "k_day_inv"]).reset_index(drop=True),
            use_container_width=True,
        )

        st.subheader("Clase de degradación por tipo de polímero")
        counts = filtered.groupby(["polymer_type", "degradation_class"]).size().reset_index(name="curvas")
        fig_bar = go.Figure()
        for cls, color in CLASS_COLORS.items():
            sub = counts[counts["degradation_class"] == cls]
            fig_bar.add_trace(go.Bar(x=sub["polymer_type"], y=sub["curvas"], name=cls, marker_color=color))
        fig_bar.update_layout(barmode="group", xaxis_title="Tipo de polímero", yaxis_title="Cantidad de curvas")
        st.plotly_chart(fig_bar, use_container_width=True)

# --- Pestaña 2: predictor interactivo ---
with tab_predict:
    st.write(
        "Ingresá las propiedades de un polímero hipotético y elegí un modelo para predecir "
        "su categoría de tasa de degradación (baja/media/alta)."
    )
    models = train_models(df_curves, cfg)

    col1, col2 = st.columns(2)
    with col1:
        polymer_type = st.selectbox("Tipo de polímero", sorted(df_curves["polymer_type"].unique()))
        medium = st.selectbox("Medio de ensayo", sorted(df_curves["medium"].unique()))
        geometry_type = st.selectbox("Geometría", sorted(df_curves["geometry_type"].unique()))
        fabrication_method = st.selectbox("Método de fabricación", sorted(df_curves["fabrication_method"].unique()))
    with col2:
        temperature_C = st.slider("Temperatura (°C)", 30.0, 45.0, 37.0, 0.5)
        pH = st.slider("pH", 2.0, 9.0, 7.4, 0.1)
        initial_mw_kDa = st.number_input(
            "Peso molecular inicial (kDa)",
            min_value=float(df_curves["initial_mw_kDa"].min()),
            max_value=float(df_curves["initial_mw_kDa"].max()),
            value=float(df_curves["initial_mw_kDa"].median()),
        )
        porosity_pct = st.slider("Porosidad (%)", 0.0, 95.0, float(df_curves["porosity_pct"].median()), 1.0)

    model_name = st.radio("Modelo", list(models.keys()), horizontal=True)

    if st.button("Predecir", type="primary"):
        subset = df_curves[df_curves["polymer_type"] == polymer_type]
        if subset.empty:
            subset = df_curves
        fractions = {c: float(subset[c].median()) for c in FRACTION_COLUMNS}

        input_row = {
            "polymer_type": polymer_type, "medium": medium,
            "geometry_type": geometry_type, "fabrication_method": fabrication_method,
            "temperature_C": temperature_C, "pH": pH,
            "initial_mw_kDa": initial_mw_kDa, "porosity_pct": porosity_pct,
            "assay_duration_days": float(df_curves["assay_duration_days"].median()),
            "n_points": float(df_curves["n_points"].median()),
            **fractions,
        }
        input_df = pd.DataFrame([input_row])[REAL_CATEGORICAL_COLUMNS + REAL_NUMERIC_COLUMNS]

        model_info = models[model_name]
        proba = model_info["pipeline"].predict_proba(input_df)[0]
        classes = model_info["classes"]
        pred_class = classes[int(np.argmax(proba))]

        st.markdown(f"### Clase predicha: **:{'green' if pred_class=='baja' else 'orange' if pred_class=='media' else 'red'}[{pred_class.upper()}]**")

        fig_proba = go.Figure(go.Bar(
            x=classes, y=proba, marker_color=[CLASS_COLORS.get(c, "gray") for c in classes],
        ))
        fig_proba.update_layout(yaxis_title="Probabilidad", yaxis_range=[0, 1], height=350)
        st.plotly_chart(fig_proba, use_container_width=True)

        with st.expander("Ver composición de copolímero usada (mediana de curvas de este tipo)"):
            st.json(fractions)
