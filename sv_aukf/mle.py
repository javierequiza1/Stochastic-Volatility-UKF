"""Estimacion MLE multistart en el espacio libre R^4.

Unifica estimate_mle_L2 / estimate_mle_L3 (antes casi identicas, salvo por
que filtro llamaban). Usa siempre el espacio no restringido -sin bounds
activos en el optimizador-, igual que idea.py/idea2.py: las biyecciones
tanh/exp garantizan el soporte correcto por construccion, asi que no hace
falta que L-BFGS-B recorte nada (a diferencia de estimate_mle_L2 en
models.py, que si usaba bounds y por tanto podia comportarse distinto).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

from .filters import RUN_AUKF
from .transforms import theta_to_unconstrained, unconstrained_to_theta

# Algunas semillas del multistart caen en zonas extremas del espacio libre
# (sigma_eta muy grande, etc.) donde exp() desborda de forma controlada -- el
# resultado ya queda descartado por los chequeos de isfinite en filters.py,
# así que silenciamos el ruido igual que hacía el models.py original.
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


def _neg_ll_free(theta_u, returns, model: str):
    theta = unconstrained_to_theta(theta_u)
    ll = RUN_AUKF[model](returns, tuple(theta))
    return -ll if np.isfinite(ll) else 1e15


def estimate_mle(returns: np.ndarray, model: str, mle_priors: dict, verbose: bool = True):
    """model: 'L2' o 'L3'. mle_priors: seccion 'mle' del JSON de priors."""
    seeds_structural = [[
        mle_priors["alpha0"], mle_priors["phi0"],
        mle_priors["sigma0"], mle_priors["rho0"],
    ]] + list(mle_priors.get("extra_seeds", []))

    best_nll, best_x_u = np.inf, None
    if verbose:
        print(f"\n[MLE-{model}] Multistart en espacio libre R^4 ({len(seeds_structural)} semillas)...")

    for x0_struct in seeds_structural:
        x0_u = theta_to_unconstrained(x0_struct)
        res = minimize(
            _neg_ll_free,
            x0=x0_u,
            args=(returns, model),
            method="L-BFGS-B",
            bounds=None,
            options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-8, "disp": False},
        )
        if verbose:
            status = "OK" if res.success else f"fallo ({res.message[:40]})"
            print(f"  negLL={res.fun:.4f}  {status}")
        if res.fun < best_nll:
            best_nll, best_x_u = res.fun, res.x

    if best_x_u is None:
        raise RuntimeError(f"[MLE-{model}] ninguna semilla convergio.")

    theta_hat = unconstrained_to_theta(best_x_u)
    if verbose:
        print(
            f"[MLE-{model}] max log-lik = {-best_nll:.2f}  |  "
            f"alpha={theta_hat[0]:.4f} phi={theta_hat[1]:.4f} "
            f"sigma_eta={theta_hat[2]:.4f} rho={theta_hat[3]:.4f}"
        )
    return theta_hat, -best_nll
