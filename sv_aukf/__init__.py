"""SV-AUKF: estimacion bayesiana de volatilidad estocastica con leverage
mediante Filtro de Kalman Unscented de estado aumentado (L=2 / L=3).

Reemplaza el conjunto anterior de scripts sueltos y duplicados
(idea.py, idea2.py, models.py, bayes_utils.py, utils.py,
run_aukf_L2.py, run_aukf_L3.py) por un unico paquete modular:

    config.py      -> rutas + hiperparametros (fuente de verdad unica)
    data_loader.py -> descarga y carga de datos (sin cambios de fondo)
    transforms.py  -> biyecciones theta <-> theta_u y jacobiano
    filters.py     -> nucleo AUKF (L=2 y L=3), calibrado consistentemente
    priors.py      -> priors bayesianos (JSON) + bounds/semillas de MLE
    mle.py         -> estimacion MLE multistart (unificada)
    mcmc.py         -> Metropolis adaptativo (unificado)
    diagnostics.py -> R-hat, ESS, MCSE, Geweke, diagnostico de innovaciones
    io_utils.py     -> exportacion + logica de cache
    pipeline.py     -> orquestador run_model('L2' | 'L3')
"""
from .data_loader import download_dataset, load_data
from .diagnostics import residual_diagnostics, summarize_mcmc
from .filters import compute_mu1, run_aukf_L2, run_aukf_L3
from .mcmc import run_mcmc
from .mle import estimate_mle
from .pipeline import run_model

__all__ = [
    "download_dataset",
    "load_data",
    "run_aukf_L2",
    "run_aukf_L3",
    "compute_mu1",
    "estimate_mle",
    "run_mcmc",
    "summarize_mcmc",
    "residual_diagnostics",
    "run_model",
]
