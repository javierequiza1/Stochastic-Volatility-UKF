"""Configuracion central del pipeline SV-AUKF.

Todos los hiperparametros numericos que antes estaban duplicados y
desincronizados entre idea.py, idea2.py y models.py viven aqui, en un
unico lugar. La fuente de verdad para (lambda, beta, gamma) es
idea.py / idea2.py: son los scripts que generaron los resultados ya
documentados en el TFM (diagnostics_L2_SV.csv, diagnostics_L3_SV.csv).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto (relativas; nada de rutas absolutas de Windows)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SVAUKF_ROOT", Path(__file__).resolve().parent.parent))
DATASET_DIR = PROJECT_ROOT / "dataset"
PRIORS_DIR = PROJECT_ROOT / "priors"
RESULTS_DIR = PROJECT_ROOT / "results"


def results_dir_for(model: str) -> Path:
    """model: 'L2' o 'L3'."""
    d = RESULTS_DIR / model
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Hiperparametros del UKF -- FUENTE DE VERDAD: idea.py / idea2.py
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UKFConfig:
    L: int
    lam: float
    beta: float

    @property
    def gamma(self) -> float:
        return (self.L + self.lam) ** 0.5


UKF_CONFIGS = {
    "L2": UKFConfig(L=2, lam=1.0, beta=2.0),  # gamma = sqrt(3)  (idea2.py, coincide con el anexo)
    "L3": UKFConfig(L=3, lam=1.0, beta=2.0),  # gamma = 2        (idea.py,  coincide con el anexo)
}

# ---------------------------------------------------------------------------
# Configuracion MCMC (Adaptive Metropolis, Haario et al. 2001)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MCMCConfig:
    n_iter: int = 50_000
    n_burn: int = 5_000
    n_chains: int = 4
    base_seed: int = 202406
    warmup: int = 1_000
    s_d: float = (2.4 ** 2) / 4  # escala optima de Gelman et al. (1996) para d=4
    epsilon_am: float = 1e-6
    c0_scale: float = 1e-3
    seed_jitter: float = 0.1  # dispersion inicial de cada cadena en torno al MLE


MCMC_CONFIG = MCMCConfig()

# ---------------------------------------------------------------------------
# Dataset por defecto
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "SPY"
DEFAULT_START = "2018-03-13"
DEFAULT_END = "2026-05-30"
