"""Motor MCMC: Metropolis adaptativo (Haario et al.) en espacio libre R^4.

Unico motor compartido entre L=2 y L=3 -- reemplaza las dos copias casi
identicas de run_adaptive_chain que habia en idea.py e idea2.py.
"""
from __future__ import annotations

import numpy as np

from .config import MCMCConfig
from .filters import RUN_AUKF
from .priors import log_prior
from .transforms import log_jacobian, theta_to_unconstrained, unconstrained_to_theta


def log_posterior_u(theta_u, returns, model: str, bayes_priors: dict) -> float:
    theta = unconstrained_to_theta(theta_u)
    ll = RUN_AUKF[model](returns, tuple(theta))
    if not np.isfinite(ll) or ll < -1e10:
        return -np.inf
    lp = log_prior(theta, bayes_priors)
    if not np.isfinite(lp):
        return -np.inf
    return float(ll + lp + log_jacobian(theta_u))


def run_adaptive_chain(
    returns, seed_theta_u, model, bayes_priors, cfg: MCMCConfig,
    iterations, burn_in, chain_id, rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    C_0 = np.eye(4) * cfg.c0_scale
    theta_u_current = np.asarray(seed_theta_u, dtype=float).copy()
    post_current = log_posterior_u(theta_u_current, returns, model, bayes_priors)

    attempts = 0
    while not np.isfinite(post_current) and attempts < 50:
        theta_u_current = seed_theta_u + rng.normal(0, 0.05, size=4)
        post_current = log_posterior_u(theta_u_current, returns, model, bayes_priors)
        attempts += 1
    if not np.isfinite(post_current):
        raise ValueError(f"Cadena #{chain_id} ({model}): no se encontro punto de partida valido.")

    print(f"  -> [{model}] Cadena #{chain_id}  |  {iterations} iters  |  burn-in={burn_in}")

    chain_u = []
    n_accepted = 0

    for t in range(iterations):
        if t < cfg.warmup or len(chain_u) < 10:
            C_t = C_0
        else:
            history = np.array(chain_u[cfg.warmup:])
            if history.shape[0] >= 5:
                cov_emp = np.cov(history.T)
                if not np.all(np.linalg.eigvalsh(cov_emp) > 0):
                    cov_emp += 1e-4 * np.eye(4)
                C_t = cfg.s_d * cov_emp + cfg.epsilon_am * np.eye(4)
            else:
                C_t = C_0

        try:
            theta_u_prop = rng.multivariate_normal(theta_u_current, C_t)
        except np.linalg.LinAlgError:
            theta_u_prop = theta_u_current + rng.normal(0, 1e-3, size=4)

        post_prop = log_posterior_u(theta_u_prop, returns, model, bayes_priors)

        if np.isfinite(post_prop):
            log_r = post_prop - post_current
            accept = (log_r > 0) or (np.log(rng.uniform()) < log_r)
        else:
            accept = False

        if accept:
            theta_u_current, post_current = theta_u_prop, post_prop
            n_accepted += 1

        chain_u.append(theta_u_current.copy())

    acc_rate = n_accepted / iterations
    print(f"     Cadena #{chain_id} ({model}) completada  |  tasa aceptacion: {acc_rate:.3f}")
    if not (0.10 < acc_rate < 0.60):
        print("     [AVISO] tasa fuera de (0.10, 0.60) -- considera ajustar C_0/warmup.")

    post_burn_u = np.array(chain_u[burn_in:])
    theta_samples = np.array([unconstrained_to_theta(x) for x in post_burn_u])
    return theta_samples, post_burn_u, acc_rate


def run_mcmc(returns, model, mle_theta, bayes_priors, cfg: MCMCConfig = MCMCConfig()):
    """Corre cfg.n_chains cadenas partiendo de jitter alrededor del MLE."""
    mle_u = theta_to_unconstrained(mle_theta)
    rng = np.random.default_rng(cfg.base_seed)

    print(
        f"\n[MCMC-{model}] {cfg.n_chains} cadenas x {cfg.n_iter} iters | "
        f"burn-in={cfg.n_burn} | seed={cfg.base_seed}"
    )

    all_theta, all_theta_u, all_acc = [], [], []
    for m in range(cfg.n_chains):
        chain_seed = int(rng.integers(0, 2**31 - 1))
        local_rng = np.random.default_rng(chain_seed)
        seed_u = mle_u + local_rng.normal(0, cfg.seed_jitter, size=4)
        theta_s, theta_u_s, acc = run_adaptive_chain(
            returns, seed_u, model, bayes_priors, cfg,
            cfg.n_iter, cfg.n_burn, chain_id=m + 1, rng=local_rng,
        )
        all_theta.append(theta_s)
        all_theta_u.append(theta_u_s)
        all_acc.append(acc)

    all_theta = np.array(all_theta)  # (M, N_post, 4)
    all_theta_u = np.array(all_theta_u)  # (M, N_post, 4)
    return all_theta, all_theta_u, np.array(all_acc)
