"""Diagnosticos MCMC (R-hat, ESS, MCSE, Geweke) y de innovaciones del filtro.

Unico modulo de diagnosticos -- reemplaza bayes_utils.py (ESS/MCSE), las
copias de compute_gelman_rubin/compute_geweke duplicadas en idea.py e
idea2.py, y utils.diagnostics() para las innovaciones (que tenia el bug
de no guardar nunca a disco al no recibir output_path desde
run_aukf_L2.py).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2, jarque_bera

from .transforms import PARAM_NAMES


# ---------------------------------------------------------------------------
# ESS / MCSE
# ---------------------------------------------------------------------------
def compute_ess(chain: np.ndarray) -> np.ndarray:
    """ESS por truncamiento de autocorrelacion (regla de Geyer)."""
    chain = np.asarray(chain, dtype=float)
    n_samples, n_params = chain.shape
    ess = np.zeros(n_params)
    for p in range(n_params):
        x = chain[:, p] - chain[:, p].mean()
        var_x = np.var(x)
        if var_x == 0.0:
            continue
        acf = np.correlate(x, x, mode="full")[n_samples - 1:] / (var_x * n_samples)
        acf = acf[1:]
        tau = 1.0
        for lag in range(1, min(len(acf), n_samples // 2) + 1):
            rho = acf[lag - 1]
            if not np.isfinite(rho) or rho <= 0.0:
                break
            tau += 2.0 * rho
        ess[p] = n_samples / max(tau, 1.0)
    return ess


def compute_mcse(chain: np.ndarray) -> np.ndarray:
    sd = chain.std(axis=0, ddof=1)
    ess = compute_ess(chain)
    mcse = np.zeros_like(sd)
    valid = ess > 0.0
    mcse[valid] = sd[valid] / np.sqrt(ess[valid])
    return mcse


# ---------------------------------------------------------------------------
# Gelman-Rubin R-hat
# ---------------------------------------------------------------------------
def compute_gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """chains: (M, N, P). R-hat < 1.01 => convergencia satisfactoria."""
    M, N, P = chains.shape
    means = chains.mean(axis=1)
    grand_mean = means.mean(axis=0)
    B = (N / (M - 1)) * np.sum((means - grand_mean) ** 2, axis=0)
    W = (1 / (M * (N - 1))) * np.sum((chains - means[:, None, :]) ** 2, axis=(0, 1))
    var_plus = ((N - 1) / N) * W + (1 / N) * B
    return np.sqrt(var_plus / W)


# ---------------------------------------------------------------------------
# Geweke
# ---------------------------------------------------------------------------
def _spectral_variance(x: np.ndarray) -> float:
    N = len(x)
    xc = x - x.mean()
    c0 = np.dot(xc, xc) / N
    M = int(np.floor(np.sqrt(N)))
    s0 = c0
    for k in range(1, M + 1):
        ck = np.dot(xc[:-k], xc[k:]) / N
        s0 += 2 * (1.0 - k / (M + 1)) * ck
    return max(s0, 1e-30)


def compute_geweke(chain: np.ndarray, frac_a: float = 0.1, frac_b: float = 0.5) -> np.ndarray:
    N, P = chain.shape
    n_a = int(frac_a * N)
    z_scores = np.zeros(P)
    for p in range(P):
        seg_a = chain[:n_a, p]
        seg_b = chain[int(frac_b * N):, p]
        s0_a, s0_b = _spectral_variance(seg_a), _spectral_variance(seg_b)
        se = np.sqrt(s0_a / len(seg_a) + s0_b / len(seg_b))
        z_scores[p] = (seg_a.mean() - seg_b.mean()) / max(se, 1e-30)
    return z_scores


def geweke_worst_over_chains(all_chains: np.ndarray) -> np.ndarray:
    """all_chains: (M, N, P). Devuelve, por parametro, el |Z| maximo entre
    cadenas (con su signo original), igual que hacian idea.py/idea2.py."""
    M = all_chains.shape[0]
    per_chain = np.array([compute_geweke(all_chains[m]) for m in range(M)])
    worst_idx = np.argmax(np.abs(per_chain), axis=0)
    return per_chain[worst_idx, np.arange(per_chain.shape[1])]


# ---------------------------------------------------------------------------
# Tabla resumen de convergencia (una fila por parametro)
# ---------------------------------------------------------------------------
def summarize_mcmc(all_theta: np.ndarray) -> pd.DataFrame:
    """all_theta: (M, N_post, 4) en espacio fisico theta."""
    combined = all_theta.reshape(-1, all_theta.shape[-1])
    r_hat = compute_gelman_rubin(all_theta)
    ess = compute_ess(combined)
    mcse = compute_mcse(combined)
    geweke = geweke_worst_over_chains(all_theta)

    report = pd.DataFrame(index=PARAM_NAMES)
    report["Post Mean"] = combined.mean(axis=0)
    report["Post SD"] = combined.std(axis=0)
    report["MCSE"] = mcse
    report["95% CI Lower"] = np.percentile(combined, 2.5, axis=0)
    report["95% CI Upper"] = np.percentile(combined, 97.5, axis=0)
    report["Gelman-Rubin R_hat"] = r_hat
    report["ESS"] = ess
    report["Geweke Max |Z|"] = geweke
    return report


# ---------------------------------------------------------------------------
# Diagnostico de innovaciones del filtro (nu_t, e_t = nu_t / sqrt(P_rr_t))
# ---------------------------------------------------------------------------
def _ljung_box(x: np.ndarray, lags: int):
    n = len(x)
    x_c = x - x.mean()
    c0 = np.dot(x_c, x_c) / n
    acf = [np.dot(x_c[:-k], x_c[k:]) / (n * c0) for k in range(1, lags + 1)]
    Q = n * (n + 2) * sum(acf[k - 1] ** 2 / (n - k) for k in range(1, lags + 1))
    return Q, 1 - chi2.cdf(Q, df=lags)


def residual_diagnostics(nu: np.ndarray, P_rr_arr: np.ndarray, output_path: str | None = None) -> str:
    """Diagnosticos sobre las innovaciones estandarizadas e_t = nu_t/sqrt(P_rr_t).

    A diferencia del utils.diagnostics() original -donde run_aukf_L2.py
    llamaba a la funcion sin output_path y por tanto nunca se guardaba
    nada a disco-, aqui el guardado se hace explicito y se llama siempre
    con una ruta desde pipeline.py.
    """
    nu_std = nu / np.sqrt(P_rr_arr)
    jb_stat, jb_p = jarque_bera(nu_std)
    lb10, p10 = _ljung_box(nu_std, 10)
    lb10_sq, p10_sq = _ljung_box(nu_std ** 2, 10)
    acf1 = np.corrcoef(nu_std[:-1], nu_std[1:])[0, 1]
    acf1_sq = np.corrcoef(nu_std[:-1] ** 2, nu_std[1:] ** 2)[0, 1]

    lines = [
        "=" * 62,
        "DIAGNOSTICOS DE INNOVACIONES ESTANDARIZADAS  e_t = nu_t / sqrt(P_rr_t)",
        "=" * 62,
        f"  Media (ideal 0):           {nu_std.mean():.6f}",
        f"  Std   (ideal 1):           {nu_std.std():.6f}",
        f"  Sesgo:                     {pd.Series(nu_std).skew():.4f}",
        f"  Curtosis exceso:           {pd.Series(nu_std).kurt():.4f}",
        f"  Jarque-Bera:               stat={jb_stat:.2f}  p={jb_p:.4f}  {'OK' if jb_p > 0.05 else 'X'}",
        f"  Ljung-Box(e,  lag=10):     Q={lb10:.2f}  p={p10:.4f}  "
        f"{'OK i.i.d.' if p10 > 0.05 else 'X autocorr'}",
        f"  Ljung-Box(e^2,lag=10):     Q={lb10_sq:.2f}  p={p10_sq:.4f}  "
        f"{'OK homoced.' if p10_sq > 0.05 else 'X vol-cluster'}",
        f"  ACF(e,  lag=1):            {acf1:.4f}   (ideal ~ 0)",
        f"  ACF(e^2,lag=1):            {acf1_sq:.4f}   (ideal ~ 0)",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[OUT] Diagnosticos guardados en: {output_path}")

    return report
