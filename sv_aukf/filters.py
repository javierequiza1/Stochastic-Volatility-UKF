"""Nucleo del filtro AUKF (Augmented Unscented Kalman Filter).

Implementacion unica y modular de ambas arquitecturas del filtro
(L=2 semi-analitico, Seccion aukf_L2 del anexo; L=3 estado totalmente
aumentado, Seccion aukf_L3), calibrada segun config.UKF_CONFIGS.

Sustituye a las tres copias previas del nucleo que existian en el
proyecto (idea.py, idea2.py, models.py), que estaban desincronizadas en
sus hiperparametros (lambda, gamma) -- ver aviso en priors.py.
"""
from __future__ import annotations

import numpy as np

from .config import UKF_CONFIGS, UKFConfig


def _ukf_weights(cfg: UKFConfig):
    n_sigma = 2 * cfg.L + 1
    wm = np.full(n_sigma, 1.0 / (2.0 * (cfg.L + cfg.lam)))
    wc = np.full(n_sigma, 1.0 / (2.0 * (cfg.L + cfg.lam)))
    wm[0] = cfg.lam / (cfg.L + cfg.lam)
    wc[0] = cfg.lam / (cfg.L + cfg.lam) + cfg.beta
    return wm, wc


_WM = {name: _ukf_weights(cfg)[0] for name, cfg in UKF_CONFIGS.items()}
_WC = {name: _ukf_weights(cfg)[1] for name, cfg in UKF_CONFIGS.items()}
_GAMMA = {name: cfg.gamma for name, cfg in UKF_CONFIGS.items()}


def compute_mu1(alpha: float, phi: float, sigma_eta: float, rho: float) -> float:
    """Correccion analitica que garantiza E[r_t] = 0."""
    denom = max(1.0 - phi ** 2, 1e-12)
    arg = alpha / 2.0 + (sigma_eta ** 2) / (8.0 * denom)
    return -0.5 * rho * sigma_eta * np.exp(arg)


def _theta_is_valid(phi, sigma_eta, rho) -> bool:
    return (-0.9999 < phi < 0.9999) and sigma_eta > 0 and (-0.9999 < rho < 0.9999)


# ---------------------------------------------------------------------------
# L = 2 -- formulacion semi-analitica (Seccion aukf_L2 del anexo)
# ---------------------------------------------------------------------------
def _sigma_points_L2(h_pred_linear, P_hh, sigma_eta, rho, gamma):
    L11 = np.sqrt(max(P_hh, 1e-12))
    L21 = (sigma_eta * rho) / L11 if L11 > 0 else 0.0
    L22 = np.sqrt(max(1.0 - L21 ** 2, 1e-12))

    X_h = np.array([
        h_pred_linear,
        h_pred_linear + gamma * L11,
        h_pred_linear,
        h_pred_linear - gamma * L11,
        h_pred_linear,
    ])
    X_eps = np.array([0.0, gamma * L21, gamma * L22, -gamma * L21, -gamma * L22])
    return X_h, X_eps


def run_aukf_L2(returns: np.ndarray, theta: tuple, return_states: bool = False):
    alpha, phi, sigma_eta, rho = theta
    if not _theta_is_valid(phi, sigma_eta, rho):
        return (-np.inf, None, None, None, None) if return_states else -np.inf

    wm, wc, gamma = _WM["L2"], _WC["L2"], _GAMMA["L2"]
    mu1 = compute_mu1(alpha, phi, sigma_eta, rho)

    N = len(returns)
    h_filt = alpha
    P_filt = sigma_eta ** 2 / max(1.0 - phi ** 2, 1e-6)
    log_lik = 0.0

    if return_states:
        h_arr, P_arr, nu_arr, P_obs_arr = (np.empty(N) for _ in range(4))

    for t in range(N):
        h_pred_linear = alpha + phi * (h_filt - alpha)
        P_hh_analitica = phi ** 2 * P_filt + sigma_eta ** 2

        X_h_t, X_eps = _sigma_points_L2(h_pred_linear, P_hh_analitica, sigma_eta, rho, gamma)
        Y_r_t = mu1 + np.exp(X_h_t / 2.0) * X_eps

        h_pred = np.dot(wm, X_h_t)
        r_pred = np.dot(wm, Y_r_t)
        dh = X_h_t - h_pred
        dr = Y_r_t - r_pred

        P_hh = np.dot(wc, dh ** 2)
        P_rr = np.dot(wc, dr ** 2)
        P_hr = np.dot(wc, dh * dr)

        nu = returns[t] - r_pred
        if P_rr <= 0 or not np.isfinite(P_rr):
            return (-np.inf, None, None, None, None) if return_states else -np.inf

        log_lik += -0.5 * (np.log(2.0 * np.pi * P_rr) + nu ** 2 / P_rr)

        K = P_hr / max(P_rr, 1e-12)
        h_filt = h_pred + K * nu
        P_filt = max(P_hh - K ** 2 * P_rr, 1e-12)

        if return_states:
            h_arr[t] = h_filt
            P_arr[t] = P_filt
            nu_arr[t] = nu
            P_obs_arr[t] = P_rr

    if return_states:
        return log_lik, h_arr, P_arr, nu_arr, P_obs_arr
    return log_lik


# ---------------------------------------------------------------------------
# L = 3 -- formulacion de estado totalmente aumentado (Seccion aukf_L3)
# ---------------------------------------------------------------------------
def run_aukf_L3(returns: np.ndarray, theta: tuple, return_states: bool = False):
    alpha, phi, sigma_eta, rho = theta
    if not _theta_is_valid(phi, sigma_eta, rho):
        return (-np.inf, None, None, None, None) if return_states else -np.inf

    wm, wc, gamma = _WM["L3"], _WC["L3"], _GAMMA["L3"]
    mu1 = compute_mu1(alpha, phi, sigma_eta, rho)

    T = len(returns)
    L = 3
    h_hat = alpha
    P_hat = sigma_eta ** 2 / max(1.0 - phi ** 2, 1e-8)
    log_lik = 0.0

    if return_states:
        h_filt, P_filt, nu_arr, P_rr_arr = (np.zeros(T) for _ in range(4))

    for t in range(T):
        r_t = returns[t]

        x_aug = np.array([h_hat, 0.0, 0.0])
        P_aug = np.diag([P_hat, sigma_eta ** 2, 1.0])

        try:
            S = np.linalg.cholesky(P_aug)
        except np.linalg.LinAlgError:
            return (-np.inf, None, None, None, None) if return_states else -np.inf

        X_sigma = np.zeros((2 * L + 1, 3))
        X_sigma[0] = x_aug
        for i in range(L):
            X_sigma[1 + i] = x_aug + gamma * S[:, i]
            X_sigma[1 + L + i] = x_aug - gamma * S[:, i]

        X_prop = alpha + phi * (X_sigma[:, 0] - alpha) + X_sigma[:, 1]
        h_pred = np.dot(wm, X_prop)

        r_pred_sigma = mu1 + np.exp(X_prop / 2.0) * (
            rho * (X_sigma[:, 1] / sigma_eta) + np.sqrt(max(0.0, 1.0 - rho ** 2)) * X_sigma[:, 2]
        )
        r_pred = np.dot(wm, r_pred_sigma)

        dh = X_prop - h_pred
        dr = r_pred_sigma - r_pred

        P_pred = np.dot(wc, dh ** 2)
        P_rr = np.dot(wc, dr ** 2)
        P_xr = np.dot(wc, dh * dr)

        nu = r_t - r_pred
        if P_rr <= 0 or not np.isfinite(P_rr):
            return (-np.inf, None, None, None, None) if return_states else -np.inf

        log_lik += -0.5 * (np.log(2.0 * np.pi * P_rr) + nu ** 2 / P_rr)

        K = P_xr / max(P_rr, 1e-12)
        h_hat = h_pred + K * nu
        P_hat = max(P_pred - (K ** 2) * P_rr, 1e-12)

        if return_states:
            h_filt[t] = h_hat
            P_filt[t] = P_hat
            nu_arr[t] = nu
            P_rr_arr[t] = P_rr

    if return_states:
        return log_lik, h_filt, P_filt, nu_arr, P_rr_arr
    return log_lik


RUN_AUKF = {"L2": run_aukf_L2, "L3": run_aukf_L3}
