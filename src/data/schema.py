"""
Esquema de datos del proyecto.

Define las columnas esperadas en el dataset consolidado, según lo
establecido en el Alcance del proyecto (Sección 4) y en Comprensión de
los datos de CRISP-DM (Sección 8.2) del plan de proyecto.

Cada curva experimental digitalizada (punto a punto, tiempo vs. % de
pérdida de masa) da lugar a una fila en el dataset "long format". Luego,
en la etapa de feature engineering, cada curva se resume en una fila
"wide format" con el parámetro cinético k ajustado.
"""

from dataclasses import dataclass, field


# --- Columnas del dataset en formato "long" (una fila = un punto de la curva) ---
LONG_FORMAT_COLUMNS = [
    "curve_id",          # identificador único de la curva/ensayo (str)
    "source_doi",        # DOI o referencia de la publicación de origen (str)
    "polymer_type",      # tipo de polímero: PLA, PCL, PGA, PLGA, etc. (str, categórica)
    "medium",            # medio de ensayo: SBF, PBS, etc. (str, categórica)
    "temperature_C",     # temperatura de ensayo en °C (float)
    "pH",                # pH del medio (float)
    "initial_mw_kDa",    # peso molecular inicial en kDa (float)
    "crystallinity_pct", # cristalinidad inicial en % (float)
    "surface_area_mm2",  # área superficial de la muestra en mm² (float)
    "time_days",         # tiempo desde el inicio del ensayo, en días (float)
    "mass_loss_pct",     # % de pérdida de masa acumulada a ese tiempo (float)
]

# --- Columnas del dataset en formato "wide" (una fila = una curva resumida) ---
WIDE_FORMAT_FEATURE_COLUMNS = [
    "polymer_type",
    "medium",
    "temperature_C",
    "pH",
    "initial_mw_kDa",
    "crystallinity_pct",
    "surface_area_mm2",
    "assay_duration_days",   # duración total del ensayo observado
    "n_points",               # cantidad de puntos temporales disponibles
]

TARGET_REGRESSION = "k_day_inv"          # constante cinética k (día^-1)
TARGET_CLASSIFICATION = "degradation_class"  # baja / media / alta

CATEGORICAL_COLUMNS = ["polymer_type", "medium"]
NUMERIC_COLUMNS = [
    "temperature_C", "pH", "initial_mw_kDa", "crystallinity_pct",
    "surface_area_mm2", "assay_duration_days", "n_points",
]


@dataclass
class ValidationResult:
    is_valid: bool
    missing_columns: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def validate_long_format(df) -> ValidationResult:
    """Valida que un DataFrame cumpla el esquema 'long format' mínimo."""
    missing = [c for c in LONG_FORMAT_COLUMNS if c not in df.columns]
    warnings = []

    if not missing:
        if df["time_days"].lt(0).any():
            warnings.append("Existen valores negativos en 'time_days'.")
        if df["mass_loss_pct"].lt(0).any() or df["mass_loss_pct"].gt(100).any():
            warnings.append("Existen valores de 'mass_loss_pct' fuera del rango [0, 100].")
        if df["pH"].lt(0).any() or df["pH"].gt(14).any():
            warnings.append("Existen valores de 'pH' fuera del rango [0, 14].")

    return ValidationResult(is_valid=(len(missing) == 0), missing_columns=missing, warnings=warnings)
