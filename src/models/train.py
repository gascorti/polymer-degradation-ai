"""
Orquestador de entrenamiento: entrena Random Forest, XGBoost y LSTM sobre el
mismo dataset, registra parámetros/métricas/artefactos en MLflow (HU3), y
opcionalmente aplica GridSearchCV para tuning de hiperparámetros (HU4).

Uso:
    python -m src.models.train --config config/config.yaml
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from src.preprocessing.features import split_dataset, get_X_y
from src.preprocessing.sequences import build_sequence_dataset
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost_model import build_xgboost_pipeline, XGBOOST_AVAILABLE
from src.models.lstm_model import build_lstm_model, train_lstm, TORCH_AVAILABLE
from src.evaluation.metrics import (
    classification_metrics, regression_metrics, build_comparison_table, classification_report_table,
)

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _mlflow_start(run_name):
    if MLFLOW_AVAILABLE:
        return mlflow.start_run(run_name=run_name)
    class _NullCtx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return _NullCtx()


def _mlflow_log_params(params: dict):
    if MLFLOW_AVAILABLE:
        mlflow.log_params(params)


def _mlflow_log_metrics(metrics: dict):
    if MLFLOW_AVAILABLE:
        loggable = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(loggable)


def train_random_forest(df_train, df_val, df_test, cfg, target_col="degradation_class"):
    X_train, y_train = get_X_y(df_train, target_col)
    X_val, y_val = get_X_y(df_val, target_col)
    X_test, y_test = get_X_y(df_test, target_col)

    rf_cfg = cfg["models"]["random_forest"]
    with _mlflow_start("random_forest"):
        pipeline = build_random_forest_pipeline(**rf_cfg)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        labels = sorted(df_train[target_col].unique())
        metrics = classification_metrics(y_test, y_pred, labels=labels)

        _mlflow_log_params(rf_cfg)
        _mlflow_log_metrics(metrics)
        if MLFLOW_AVAILABLE:
            mlflow.sklearn.log_model(pipeline, "model")

    return pipeline, metrics


def train_xgboost(df_train, df_val, df_test, cfg, target_col="degradation_class"):
    if not XGBOOST_AVAILABLE:
        print("[SKIP] XGBoost — el paquete 'xgboost' no está instalado en este entorno "
              "(ejecutá `pip install xgboost` y, si estás en Colab, reiniciá el entorno de ejecución).")
        return None, None

    X_train, y_train = get_X_y(df_train, target_col)
    X_test, y_test = get_X_y(df_test, target_col)

    # XGBoost requiere target numérico
    classes = sorted(df_train[target_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_num = y_train.map(class_to_idx)
    y_test_num = y_test.map(class_to_idx)

    xgb_cfg = cfg["models"]["xgboost"]
    try:
        with _mlflow_start("xgboost"):
            pipeline = build_xgboost_pipeline(**xgb_cfg)
            pipeline.fit(X_train, y_train_num)

            y_pred_num = pipeline.predict(X_test)
            metrics = classification_metrics(y_test_num, y_pred_num, labels=list(range(len(classes))))

            _mlflow_log_params(xgb_cfg)
            _mlflow_log_metrics(metrics)
            if MLFLOW_AVAILABLE:
                mlflow.sklearn.log_model(
                    pipeline, "model",
                    skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
                )
    except Exception as e:
        print(f"[ERROR] XGBoost — falló durante el entrenamiento: {type(e).__name__}: {e}")
        return None, None

    return pipeline, metrics


def train_lstm_model(df_long, df_curves_train, df_curves_val, df_curves_test, cfg):
    if not TORCH_AVAILABLE:
        print("[SKIP] LSTM — el paquete 'torch' no está instalado en este entorno "
              "(ejecutá `pip install torch` y, si estás en Colab, reiniciá el entorno de ejecución).")
        return None, None

    lstm_cfg = cfg["models"]["lstm"]
    seq_len = lstm_cfg["sequence_length"]

    try:
        X_train, ids_train = build_sequence_dataset(df_long, df_curves_train, seq_len)
        X_val, ids_val = build_sequence_dataset(df_long, df_curves_val, seq_len)
        X_test, ids_test = build_sequence_dataset(df_long, df_curves_test, seq_len)

        y_train = df_curves_train.set_index("curve_id").loc[ids_train, "k_day_inv"].values
        y_val = df_curves_val.set_index("curve_id").loc[ids_val, "k_day_inv"].values
        y_test = df_curves_test.set_index("curve_id").loc[ids_test, "k_day_inv"].values

        with _mlflow_start("lstm"):
            model = build_lstm_model(sequence_length=seq_len, units=lstm_cfg["units"])
            train_lstm(
                model, X_train, y_train, X_val, y_val,
                epochs=lstm_cfg["epochs"], batch_size=lstm_cfg["batch_size"],
                random_state=lstm_cfg["random_state"],
            )
            y_pred = model.predict(X_test, verbose=0).flatten()
            metrics = regression_metrics(y_test, y_pred)

            _mlflow_log_params(lstm_cfg)
            _mlflow_log_metrics(metrics)
    except Exception as e:
        print(f"[ERROR] LSTM — falló durante el entrenamiento: {type(e).__name__}: {e}")
        return None, None

    return model, metrics


def run_full_training(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    df_curves = pd.read_csv(cfg["paths"]["processed_data"])
    df_long = pd.read_csv("data/processed/curvas_estandarizadas.csv")

    split_cfg = cfg["split"]
    df_train, df_val, df_test = split_dataset(
        df_curves, target_col="degradation_class",
        test_size=split_cfg["test_size"], val_size=split_cfg["val_size"],
        random_state=split_cfg["random_state"],
    )

    results = {}
    all_model_names = ["Random Forest", "XGBoost", "LSTM (regresión de k)"]

    print("\n--- Entrenando Random Forest ---")
    rf_model, rf_metrics = train_random_forest(df_train, df_val, df_test, cfg)
    if rf_metrics:
        results["Random Forest"] = {"accuracy": rf_metrics["accuracy"], "f1_macro": rf_metrics["f1_macro"]}

    print("\n--- Entrenando XGBoost ---")
    xgb_model, xgb_metrics = train_xgboost(df_train, df_val, df_test, cfg)
    if xgb_metrics:
        results["XGBoost"] = {"accuracy": xgb_metrics["accuracy"], "f1_macro": xgb_metrics["f1_macro"]}

    print("\n--- Entrenando LSTM ---")
    lstm_model, lstm_metrics = train_lstm_model(df_long, df_train, df_val, df_test, cfg)
    if lstm_metrics:
        results["LSTM (regresión de k)"] = lstm_metrics

    trained = list(results.keys())
    skipped = [name for name in all_model_names if name not in trained]
    print("\n=== Resumen de entrenamiento ===")
    print(f"Entrenados exitosamente ({len(trained)}/{len(all_model_names)}): {trained if trained else 'ninguno'}")
    if skipped:
        print(f"NO entrenados: {skipped}  <- revisá los mensajes [SKIP]/[ERROR] de arriba para saber por qué.")

    if results:
        comparison_df = build_comparison_table(results)
        os.makedirs(cfg["paths"]["reports"], exist_ok=True)
        comparison_path = os.path.join(cfg["paths"]["reports"], "comparacion_modelos.csv")
        comparison_df.to_csv(comparison_path)
        print("\n=== Cuadro comparativo de modelos ===")
        print(comparison_df)
        print(f"\nGuardado en {comparison_path}")

    target_col = "degradation_class"
    labels = cfg["data"]["degradation_class_labels"]
    rf_pred = xgb_pred = lstm_pred_k = None
    rf_eval = xgb_eval = None

    if rf_model is not None:
        X_test, y_test = get_X_y(df_test, target_col)
        y_pred = rf_model.predict(X_test)
        rf_pred = pd.Series(y_pred, index=df_test["curve_id"].values)
        rf_eval = {
            "y_true": y_test.values, "y_pred": y_pred,
            "y_proba": rf_model.predict_proba(X_test), "classes": list(rf_model.classes_),
            "report": classification_report_table(y_test.values, y_pred, labels),
        }

    if xgb_model is not None:
        classes = sorted(df_train[target_col].unique())
        idx_to_class = dict(enumerate(classes))
        X_test, y_test = get_X_y(df_test, target_col)
        xgb_pred_num = xgb_model.predict(X_test)
        y_pred = np.array([idx_to_class[i] for i in xgb_pred_num])
        xgb_pred = pd.Series(y_pred, index=df_test["curve_id"].values)
        xgb_eval = {
            "y_true": y_test.values, "y_pred": y_pred,
            "y_proba": xgb_model.predict_proba(X_test), "classes": classes,
            "report": classification_report_table(y_test.values, y_pred, labels),
        }

    if lstm_model is not None:
        seq_len = cfg["models"]["lstm"]["sequence_length"]
        X_test_seq, ids_test_seq = build_sequence_dataset(df_long, df_test, seq_len)
        lstm_pred_k = pd.Series(lstm_model.predict(X_test_seq, verbose=0).flatten(), index=ids_test_seq)

    if rf_eval is not None:
        rf_eval["report"].to_csv(os.path.join(cfg["paths"]["reports"], "reporte_clasificacion_random_forest.csv"))
    if xgb_eval is not None:
        xgb_eval["report"].to_csv(os.path.join(cfg["paths"]["reports"], "reporte_clasificacion_xgboost.csv"))

    extras = {
        "df_test": df_test,
        "df_long": df_long,
        "labels": labels,
        "rf_pred": rf_pred,
        "xgb_pred": xgb_pred,
        "rf_eval": rf_eval,
        "xgb_eval": xgb_eval,
        "lstm_pred_k": lstm_pred_k,
    }

    return results, extras


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    run_full_training(args.config)
