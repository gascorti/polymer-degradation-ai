"""
Generador de datos sintéticos.

Mientras se completa la extracción digital real de curvas desde
publicaciones científicas (HU1, Sprint 1-2), este módulo genera un
dataset sintético pero fisicoquímicamente plausible, que permite
desarrollar y validar el resto del pipeline (HU2 a HU6) de punta a punta.

La cinética utilizada es de pseudo-primer orden:
    M(t) = M0 * exp(-k * t)
donde k depende de las propiedades del material y del ensayo mediante
una relación log-lineal con ruido, inspirada en la literatura de
degradación hidrolítica de poliésteres alifáticos (PLA, PCL, PGA, PLGA).

Uso:
    python -m src.data.generate_synthetic --n_curves 60 --seed 42
"""

import argparse
import numpy as np
import pandas as pd

RNG_DEFAULT_SEED = 42

# Rangos plausibles por tipo de polímero (valores orientativos de literatura)
POLYMER_PROFILES = {
    "PGA":  {"base_log_k": -3.2, "mw_range": (20, 80),  "xtal_range": (30, 55)},
    "PLGA": {"base_log_k": -3.8, "mw_range": (30, 150), "xtal_range": (0, 20)},
    "PLA":  {"base_log_k": -5.0, "mw_range": (50, 300), "xtal_range": (10, 50)},
    "PCL":  {"base_log_k": -6.2, "mw_range": (40, 200), "xtal_range": (40, 70)},
    "PHB":  {"base_log_k": -5.6, "mw_range": (100, 400),"xtal_range": (50, 80)},
}

MEDIA = ["SBF", "PBS", "Agua destilada", "Suero fisiológico"]


def _simulate_curve(curve_id, polymer_type, rng):
    profile = POLYMER_PROFILES[polymer_type]

    medium = rng.choice(MEDIA)
    temperature_C = float(np.round(rng.normal(37, 3), 1))         # ~ condiciones fisiológicas
    pH = float(np.round(rng.normal(7.3, 0.5), 2))
    initial_mw_kDa = float(np.round(rng.uniform(*profile["mw_range"]), 1))
    crystallinity_pct = float(np.round(rng.uniform(*profile["xtal_range"]), 1))
    surface_area_mm2 = float(np.round(rng.uniform(20, 300), 1))

    # Relación log-lineal de k con las variables (con ruido), a partir de un
    # perfil base por tipo de polímero. Mayor temperatura, pH más ácido,
    # menor peso molecular, menor cristalinidad y mayor área -> mayor k.
    log_k = (
        profile["base_log_k"]
        + 0.05 * (temperature_C - 37)
        - 0.15 * (pH - 7.3)
        - 0.004 * (initial_mw_kDa - np.mean(profile["mw_range"]))
        - 0.01 * (crystallinity_pct - np.mean(profile["xtal_range"]))
        + 0.002 * (surface_area_mm2 - 150)
        + rng.normal(0, 0.25)
    )
    k = float(np.exp(log_k))  # día^-1

    assay_duration_days = float(rng.choice([30, 60, 90, 120, 180, 270, 365]))
    n_points = int(rng.integers(6, 14))
    time_days = np.sort(rng.uniform(0, assay_duration_days, size=n_points))
    time_days[0] = 0.0

    noise = rng.normal(0, 1.5, size=n_points)
    mass_loss_pct = 100 * (1 - np.exp(-k * time_days)) + noise
    mass_loss_pct = np.clip(mass_loss_pct, 0, 100)
    mass_loss_pct = np.maximum.accumulate(mass_loss_pct)  # monotonía físicamente esperable

    rows = []
    for t, m in zip(time_days, mass_loss_pct):
        rows.append({
            "curve_id": curve_id,
            "source_doi": f"10.synthetic/{curve_id}",
            "polymer_type": polymer_type,
            "medium": medium,
            "temperature_C": temperature_C,
            "pH": pH,
            "initial_mw_kDa": initial_mw_kDa,
            "crystallinity_pct": crystallinity_pct,
            "surface_area_mm2": surface_area_mm2,
            "time_days": round(float(t), 2),
            "mass_loss_pct": round(float(m), 2),
        })
    return rows, k


def generate_dataset(n_curves: int = 60, seed: int = RNG_DEFAULT_SEED):
    """Genera un dataset sintético en formato long, y devuelve también los k reales (para debug)."""
    rng = np.random.default_rng(seed)
    polymer_types = list(POLYMER_PROFILES.keys())

    all_rows = []
    true_k = {}
    for i in range(n_curves):
        polymer_type = rng.choice(polymer_types)
        curve_id = f"curve_{i:03d}"
        rows, k = _simulate_curve(curve_id, polymer_type, rng)
        all_rows.extend(rows)
        true_k[curve_id] = k

    df = pd.DataFrame(all_rows)
    return df, true_k


def main():
    parser = argparse.ArgumentParser(description="Genera dataset sintético de curvas de degradación.")
    parser.add_argument("--n_curves", type=int, default=60)
    parser.add_argument("--seed", type=int, default=RNG_DEFAULT_SEED)
    parser.add_argument("--output", type=str, default="data/raw/curvas_sinteticas.csv")
    args = parser.parse_args()

    df, _ = generate_dataset(n_curves=args.n_curves, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Dataset sintético generado: {len(df)} puntos de {df['curve_id'].nunique()} curvas -> {args.output}")


if __name__ == "__main__":
    main()
