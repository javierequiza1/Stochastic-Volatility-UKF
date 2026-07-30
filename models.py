import warnings
import numpy as np
from scipy.optimize import minimize

# Silenciar avisos durante la optimización MLE
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


def compute_mu1(alpha: float, phi: float, sigma_eta: float, rho: float) -> float:
    """Corrección analítica para garantizar E[r_t] = 0."""
    denom = max(1.0 - phi**2, 1e-12)
    arg = alpha / 2.0 + (sigma_eta**2) / (8.0 * denom)
    return -0.5 * rho * sigma_eta * np.exp(arg)


# ==============================================================================
# ARQUITECTURA L = 2
# ==============================================================================

def sigma_points_L2(h_pred_linear: float, P_hh: float, sigma_eta: float, rho: float, gamma: float):
    """Descomposición de Cholesky analítica acoplada 2x2 para L=2."""
    L11 = np.sqrt(max(P_hh, 1e-12))
    L21 = (sigma_eta * rho) / L11 if L11 > 0 else 0.0
    L22 = np.sqrt(max(1.0 - L21**2, 1e-12))

    X_h = np.array([
        h_pred_linear,
        h_pred_linear + gamma * L11,
        h_pred_linear,
        h_pred_linear - gamma * L11,
        h_pred_linear
    ])
    X_eps = np.array([
        0.0,
        gamma * L21,
        gamma * L22,
        -gamma * L21,
        -gamma * L22
    ])

    return X_h, X_eps


def run_aukf_L2(returns: np.ndarray, theta: tuple, return_states: bool = False):
    """Filtro Unscented Kalman AUKF con arquitectura L=2."""
    alpha, phi, sigma_eta, rho = theta
    N = len(returns)

    # Parámetros UKF
    L = 2
    lambda_ukf = 2.0
    beta_ukf = 2.0
    gamma = np.sqrt(L + lambda_ukf)

    # Pesos escalados directamente con lambda y beta
    wm = np.full(2 * L + 1, 1.0 / (2.0 * (L + lambda_ukf)))
    wc = np.full(2 * L + 1, 1.0 / (2.0 * (L + lambda_ukf)))
    wm[0] = lambda_ukf / (L + lambda_ukf)
    wc[0] = lambda_ukf / (L + lambda_ukf) + beta_ukf

    mu1 = compute_mu1(alpha, phi, sigma_eta, rho)

    # Inicialización estacionaria
    h_filt = alpha
    P_filt = sigma_eta**2 / max(1.0 - phi**2, 1e-6)

    log_lik = 0.0

    if return_states:
        h_arr = np.empty(N)
        P_arr = np.empty(N)
        nu_arr = np.empty(N)
        P_obs_arr = np.empty(N)

    Y_r_t = np.zeros(2 * L + 1)

    for t in range(N):
        # 1. Predicción lineal
        h_pred_linear = alpha + phi * (h_filt - alpha)
        P_hh_analitica = phi**2 * P_filt + sigma_eta**2

        # 2. Puntos Sigma
        X_h_t, X_eps = sigma_points_L2(h_pred_linear, P_hh_analitica, sigma_eta, rho, gamma)

        # 3. Propagación no lineal
        for i in range(2 * L + 1):
            Y_r_t[i] = mu1 + np.exp(X_h_t[i] / 2.0) * X_eps[i]

        # 4. Recombinación de momentos
        h_pred = np.dot(wm, X_h_t)
        r_pred = np.dot(wm, Y_r_t)

        dh = X_h_t - h_pred
        dr = Y_r_t - r_pred

        P_hh = np.dot(wc, dh**2)
        P_rr = np.dot(wc, dr**2)
        P_hr = np.dot(wc, dh * dr)

        # 5. Innovación y actualización
        nu = returns[t] - r_pred
        if P_rr <= 0 or not np.isfinite(P_rr):
            return (-1e15, None, None, None, None) if return_states else -1e15

        log_lik += -0.5 * (np.log(2.0 * np.pi * P_rr) + nu**2 / P_rr)

        K = P_hr / max(P_rr, 1e-12)
        h_filt = h_pred + K * nu
        P_filt = max(P_hh - K**2 * P_rr, 1e-12)

        if return_states:
            h_arr[t] = h_filt
            P_arr[t] = P_filt
            nu_arr[t] = nu
            P_obs_arr[t] = P_rr

    if return_states:
        return log_lik, h_arr, P_arr, nu_arr, P_obs_arr
    return log_lik


# ==============================================================================
# ARQUITECTURA L = 3
# ==============================================================================

def run_aukf_L3(returns: np.ndarray, theta: tuple, return_states: bool = False):
    """Filtro Unscented Kalman AUKF con arquitectura aumentada L=3 (7 puntos sigma)."""
    alpha, phi, sigma_eta, rho = theta
    T = len(returns)

    # Parámetros UKF
    L = 3
    lambda_ukf = 2.0
    beta_ukf = 2.0
    gamma = np.sqrt(L + lambda_ukf)

    # Pesos escalados directamente con lambda y beta
    wm = np.full(2 * L + 1, 1.0 / (2.0 * (L + lambda_ukf)))
    wc = np.full(2 * L + 1, 1.0 / (2.0 * (L + lambda_ukf)))
    wm[0] = lambda_ukf / (L + lambda_ukf)
    wc[0] = lambda_ukf / (L + lambda_ukf) + beta_ukf

    h_hat = alpha
    P_hat = sigma_eta**2 / max(1.0 - phi**2, 1e-8)

    log_lik = 0.0
    if return_states:
        h_filt = np.zeros(T)
        P_filt = np.zeros(T)
        nu_arr = np.zeros(T)
        P_rr_arr = np.zeros(T)

    mu1_val = compute_mu1(alpha, phi, sigma_eta, rho)

    for t in range(T):
        r_t = returns[t]

        x_aug = np.array([h_hat, 0.0, 0.0])
        P_aug = np.diag([P_hat, sigma_eta**2, 1.0])

        try:
            S = np.linalg.cholesky(P_aug)
        except np.linalg.LinAlgError:
            return (-1e15, None, None, None, None) if return_states else -1e15

        X_sigma = np.zeros((2 * L + 1, 3))
        X_sigma[0] = x_aug
        for i in range(L):
            X_sigma[1 + i]     = x_aug + gamma * S[:, i]
            X_sigma[1 + L + i] = x_aug - gamma * S[:, i]

        X_prop = alpha + phi * (X_sigma[:, 0] - alpha) + X_sigma[:, 1]
        h_pred = np.sum(wm * X_prop)

        r_pred_sigma = mu1_val + np.exp(X_prop / 2.0) * (
            rho * (X_sigma[:, 1] / sigma_eta) + np.sqrt(max(0.0, 1.0 - rho**2)) * X_sigma[:, 2]
        )

        r_pred = np.sum(wm * r_pred_sigma)

        dh = X_prop - h_pred
        dr = r_pred_sigma - r_pred

        P_pred = np.sum(wc * dh**2)
        P_rr   = np.sum(wc * dr**2)
        P_xr   = np.sum(wc * dh * dr)

        nu = r_t - r_pred
        if P_rr <= 0 or not np.isfinite(P_rr):
            return (-1e15, None, None, None, None) if return_states else -1e15

        K = P_xr / max(P_rr, 1e-12)
        h_hat = h_pred + K * nu
        P_hat = max(P_pred - (K**2) * P_rr, 1e-12)

        log_lik += -0.5 * (np.log(2.0 * np.pi * P_rr) + nu**2 / P_rr)

        if return_states:
            h_filt[t] = h_hat
            P_filt[t] = P_hat
            nu_arr[t] = nu
            P_rr_arr[t] = P_rr

    if return_states:
        return log_lik, h_filt, P_filt, nu_arr, P_rr_arr
    return log_lik


# ==============================================================================
# ESTIMACIÓN MLE MULTI-START
# ==============================================================================

def estimate_mle_L2(returns: np.ndarray, priors: dict):
    b = priors["bounds"]
    bounds = [tuple(b["alpha"]), tuple(b["phi"]), tuple(b["sigma"]), tuple(b["rho"])]
    alpha0, phi0, sigma0, rho0 = priors["alpha0"], priors["phi0"], priors["sigma0"], priors["rho0"]

    seeds = [
        [alpha0,      phi0,   sigma0,       rho0      ],
        [alpha0-0.5,  phi0,   sigma0*1.5,   rho0      ],
        [alpha0+0.5,  0.95,   sigma0*0.8,   rho0-0.1  ],
        [alpha0,      0.99,   sigma0*0.5,   rho0      ],
        [alpha0-1.0,  phi0,   sigma0*2.0,   -0.15     ],
        [alpha0,      0.90,   sigma0*1.2,   -0.05     ],
    ]

    best_result, best_nll = None, np.inf

    def neg_ll(theta):
        ll = run_aukf_L2(returns, theta)
        return -ll if np.isfinite(ll) else 1e15

    print("\n[MLE] Iniciando estimación multi-start (L-BFGS-B) con arquitectura L=2...")
    for x0 in seeds:
        res = minimize(
            neg_ll,
            x0      = x0,
            method  = "L-BFGS-B",
            bounds  = bounds,
            options = {"maxiter": 2000, "ftol": 1e-9, "disp": False}
        )
        if res.fun < best_nll:
            best_nll    = res.fun
            best_result = res

    a, p, s, r = best_result.x
    return a, p, s, r, compute_mu1(a, p, s, r), -best_nll


def estimate_mle_L3(returns: np.ndarray, priors: dict):
    b = priors["bounds"]
    bounds = [tuple(b["alpha"]), tuple(b["phi"]), tuple(b["sigma"]), tuple(b["rho"])]
    alpha0, phi0, sigma0, rho0 = priors["alpha0"], priors["phi0"], priors["sigma0"], priors["rho0"]

    seeds = [
        [alpha0,      phi0,   sigma0,       rho0      ],
        [alpha0-0.5,  phi0,   sigma0*1.5,   rho0      ],
        [alpha0+0.5,  0.95,   sigma0*0.8,   rho0-0.1  ],
        [alpha0,      0.99,   sigma0*0.5,   rho0      ],
        [alpha0-1.0,  phi0,   sigma0*2.0,   -0.15     ],
        [alpha0,      0.90,   sigma0*1.2,   -0.05     ],
    ]

    best_result, best_nll = None, np.inf

    def neg_ll(theta):
        ll = run_aukf_L3(returns, theta)
        return -ll if np.isfinite(ll) else 1e15

    print("\n[MLE] Iniciando estimación multi-start (L-BFGS-B) con arquitectura L=3...")
    for x0 in seeds:
        res = minimize(
            neg_ll,
            x0      = x0,
            method  = "L-BFGS-B",
            bounds  = bounds,
            options = {"maxiter": 2000, "ftol": 1e-9, "disp": False}
        )
        if res.fun < best_nll:
            best_nll    = res.fun
            best_result = res

    a, p, s, r = best_result.x
    return a, p, s, r, compute_mu1(a, p, s, r), -best_nll