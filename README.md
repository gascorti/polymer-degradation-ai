# Aplicación de IA para la predicción de tasas de degradación de polímeros biocompatibles

Trabajo Final — Carrera de Especialización en Inteligencia Artificial (FIUBA)
Autor: Lic. Gastón Corti · Director: Esp. Ing. Ariadna Garmendia

Implementación del pipeline descripto en el plan de proyecto: extracción y
estandarización de curvas de degradación (pérdida de masa vs. tiempo),
ajuste cinético (constante *k*), y modelos supervisados (Random Forest,
XGBoost, LSTM) para clasificar y predecir la tasa de degradación de
polímeros biocompatibles en medios fisiológicos.

## Estado actual

Este repo ya corre **de punta a punta con datos sintéticos** (generados con
una cinética de pseudo-primer orden fisicoquímicamente plausible), mientras
avanza la extracción real de curvas desde publicaciones (HU1, Épica 1). Esto
permite desarrollar y validar todo el resto del pipeline (preprocesamiento,
modelado, evaluación, visualización) sin esperar a tener el dataset real
completo. Cuando la extracción bibliográfica esté lista, alcanza con
reemplazar `data/raw/curvas_sinteticas.csv` por el CSV real (mismo esquema,
ver `src/data/schema.py`) y correr `main.py` de nuevo.

## Estructura del proyecto

```
polymer-degradation-ai/
├── config/
│   └── config.yaml            # configuración central (rutas, hiperparámetros, splits)
├── data/
│   ├── raw/                   # curvas crudas (una fila = un punto tiempo/pérdida de masa)
│   ├── processed/             # dataset estandarizado y dataset a nivel de curva (con k)
│   └── external/              # metadatos bibliográficos (DOIs, criterios de inclusión, etc.)
├── src/
│   ├── data/
│   │   ├── schema.py           # esquema de columnas + validación
│   │   └── generate_synthetic.py  # generador de datos sintéticos (dev/testing)
│   ├── preprocessing/
│   │   ├── standardize.py      # limpieza, estandarización de unidades (HU2)
│   │   ├── kinetics.py         # ajuste de k por curva + categorización baja/media/alta
│   │   ├── sequences.py        # interpolación de curvas para el modelo LSTM
│   │   └── features.py         # encoding, split train/val/test
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lstm_model.py
│   │   └── train.py            # orquestador de entrenamiento + MLflow (HU3, HU4)
│   ├── evaluation/
│   │   └── metrics.py          # métricas y cuadro comparativo (HU5)
│   └── visualization/
│       └── plots.py            # EDA y explicabilidad (HU6)
├── tests/
│   └── test_pipeline.py        # tests unitarios de cada etapa
├── outputs/
│   ├── figures/                # gráficos generados
│   └── reports/                # comparación de modelos, etc.
├── main.py                     # corre el pipeline completo (CRISP-DM end-to-end)
└── requirements.txt
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> XGBoost, TensorFlow y MLflow son opcionales: si no están instalados, el
> pipeline los omite automáticamente (con un aviso) y sigue funcionando con
> Random Forest. Instalalos para tener los 3 modelos y el tracking completo.

## Trabajar en VS Code

El repo ya incluye configuración lista en `.vscode/`:

- **`settings.json`**: apunta al intérprete de `venv/`, activa pytest como test runner y formateo automático con `black` al guardar.
- **`launch.json`**: configuraciones de debug (con breakpoints) para `main.py`, para `src.models.train` solo, y para correr los tests con el depurador.
- **`tasks.json`**: tareas rápidas desde la paleta de comandos (`Ctrl+Shift+P` / `Cmd+Shift+P` → *Tasks: Run Task*): crear el venv, instalar dependencias, correr el pipeline, correr tests, o levantar MLflow UI.
- **`extensions.json`**: extensiones recomendadas (Python, Pylance, Black, Jupyter, GitLens).

Pasos para arrancar:

1. Abrí la carpeta del proyecto en VS Code.
2. Cuando te lo sugiera, instalá las extensiones recomendadas.
3. `Ctrl+Shift+P` → *Tasks: Run Task* → **"1. Crear entorno virtual"**, después **"2. Instalar dependencias"**.
4. Seleccioná el intérprete `venv` (VS Code lo va a detectar automáticamente; si no, `Ctrl+Shift+P` → *Python: Select Interpreter*).
5. Para correr con breakpoints: pestaña **Run and Debug** → elegí *"Python: main.py (pipeline completo)"* → `F5`.
6. Para tests: pestaña **Testing** (ícono del matraz) → se descubren automáticamente los tests de `tests/test_pipeline.py`.

El repo ya tiene `git init` con un `.gitignore` que excluye datos generados, `venv/`, cachés y corridas de MLflow — solo versiona código y configuración.

## Uso rápido

Correr todo el pipeline (genera datos sintéticos, estandariza, ajusta
cinética, entrena y compara modelos, genera figuras):

```bash
python main.py --n_curves 80 --seed 42
```

Correr solo el entrenamiento (si ya existe `data/processed/dataset_estandarizado.csv`):

```bash
python -m src.models.train --config config/config.yaml
```

Ver el tracking de experimentos en MLflow (si está instalado):

```bash
mlflow ui --backend-store-uri mlruns
```

Correr los tests:

```bash
pip install pytest
pytest tests/ -v
```

## Cómo incorporar datos reales (HU1)

1. Extraer curvas con WebPlotDigitizer desde las publicaciones relevadas.
2. Volcar cada curva a filas con las columnas de `LONG_FORMAT_COLUMNS`
   (ver `src/data/schema.py`): `curve_id`, `source_doi`, `polymer_type`,
   `medium`, `temperature_C`, `pH`, `initial_mw_kDa`, `crystallinity_pct`,
   `surface_area_mm2`, `time_days`, `mass_loss_pct`.
3. Guardar el CSV en `data/raw/` (podés tener varios archivos, uno por
   lote de publicaciones relevadas).
4. Ejecutar `main.py --no-synthetic` apuntando `standardize.run_standardization_pipeline`
   a la lista de archivos reales (ver `src/preprocessing/standardize.py`,
   función `run_standardization_pipeline`).

## Métricas de éxito (Sección 8, CRISP-DM)

- Clasificación (baja/media/alta): F1-score y accuracy ≥ 0.8.
- Regresión de *k*: RMSE minimizado (LSTM).

## Notas de gobernanza y ética (Sección 12 del plan)

El pipeline trabaja exclusivamente con datos secundarios de publicaciones
científicas (no hay datos personales ni clínicos). Los resultados de este
PoC no deben usarse para decisiones de diseño de implantes sin validación
in vitro/in vivo adicional.
