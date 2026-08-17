"""Punto de entrada: pipeline completo para el modelo de estado totalmente
aumentado L=3.

Si results/L3/posterior_samples_L3_SV.csv ya existe, se reutiliza esa
estimacion (MLE+MCMC no se recalculan); en caso contrario se estima desde
cero. En ambos casos se corre el filtro final y se exportan los estados
filtrados y los diagnosticos de innovaciones.

Uso:
    python run_L3.py                  # usa cache si existe
    python run_L3.py --force          # fuerza recalcular MLE+MCMC
"""
import argparse
import os

from sv_aukf.config import DATASET_DIR, DEFAULT_END, DEFAULT_START, DEFAULT_TICKER
from sv_aukf.pipeline import run_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline SV-AUKF L=3")
    parser.add_argument("--force", action="store_true", help="Ignora la cache y recalcula MLE+MCMC.")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    args = parser.parse_args()

    dataset_path = os.path.join(DATASET_DIR, f"dataset_{args.ticker}_daily.csv")

    result = run_model(
        "L3",
        dataset_path=dataset_path,
        start_date=args.start,
        end_date=args.end,
        force_recompute=args.force,
    )
    print(f"\n[DONE] L3 -- resultados en: {result['results_dir']}")
