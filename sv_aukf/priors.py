"""Priors bayesianos y bounds/semillas de MLE.

Fuente unica en JSON (priors/priors_L2.json, priors/priors_L3.json) para
evitar la desincronizacion que habia entre las constantes PRIOR_*
hardcodeadas en idea.py/idea2.py y el priors_L2.json que solo usaba
models.py con otro esquema.

*** AVISO IMPORTANTE ***
Los valores de la seccion "bayes" en priors_L2.json / priors_L3.json
replican las constantes PRIOR_* que efectivamente uso el MCMC
(idea.py/idea2.py) para generar los resultados ya documentados en el TFM:
  sigma_alpha = 3.0, sigma_phi = 0.15, sigma_rho = 0.40.

Estos NO coinciden con los valores redactados en la Seccion "Eleccion de
las distribuciones a priori" del anexo LaTeX:
  sigma_alpha = 2.0, sigma_phi = 0.10, sigma_rho = 0.30.

Hay que decidir cual es la version correcta y alinear texto <-> codigo
antes de dar por definitivos los resultados (ver conversacion).
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import halfnorm, norm, truncnorm


def load_priors(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encuentra el archivo de priors: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


def log_prior(theta, bayes_priors: dict) -> float:
    """Log-densidad del prior sobre el espacio fisico theta=(alpha,phi,sigma_eta,rho).

    bayes_priors: diccionario con las claves alpha_loc, alpha_scale,
    phi_loc, phi_scale, sigma_scale, rho_loc, rho_scale (ver seccion
    "bayes" de priors_L2.json / priors_L3.json).
    """
    alpha, phi, sigma_eta, rho = np.asarray(theta, dtype=float)

    if not np.isfinite([alpha, phi, sigma_eta, rho]).all():
        return -np.inf
    if not (-1.0 < phi < 1.0) or sigma_eta <= 0.0 or not (-1.0 < rho < 1.0):
        return -np.inf

    p = bayes_priors
    lp = norm.logpdf(alpha, loc=p["alpha_loc"], scale=p["alpha_scale"])
    lp += truncnorm.logpdf(
        phi,
        a=(-1.0 - p["phi_loc"]) / p["phi_scale"],
        b=(1.0 - p["phi_loc"]) / p["phi_scale"],
        loc=p["phi_loc"],
        scale=p["phi_scale"],
    )
    lp += halfnorm.logpdf(sigma_eta, scale=p["sigma_scale"])
    lp += truncnorm.logpdf(
        rho,
        a=(-1.0 - p["rho_loc"]) / p["rho_scale"],
        b=(1.0 - p["rho_loc"]) / p["rho_scale"],
        loc=p["rho_loc"],
        scale=p["rho_scale"],
    )
    return float(lp)
