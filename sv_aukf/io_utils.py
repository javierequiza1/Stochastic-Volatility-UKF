"""Exportacion unificada de resultados y logica de cache.

Reemplaza:
 - bayes_utils.export_model_outputs (no incluia estados filtrados)
 - los bloques de exportacion manual duplicados en idea.py / idea2.py
 - el guardado incompleto de run_aukf_L2.py / run_aukf_L3.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .transforms import PARAM_NAMES, PARAM_NAMES_U


def posterior_samples_path(results_dir, model: str) -> str:
    return os.path.join(results_dir, f"posterior_samples_{model}_SV.csv")


def has_cached_posterior(results_dir, model: str) -> bool:
    """Criterio de cache acordado: si el CSV de posterior_samples existe,
    se reutiliza tal cual (sin comprobar hash de datos/priors)."""
    return os.path.exists(posterior_samples_path(results_dir, model))


def load_cached_posterior(results_dir, model: str) -> np.ndarray:
    path = posterior_samples_path(results_dir, model)
    df = pd.read_csv(path)
    return df[PARAM_NAMES].values


def export_mcmc_outputs(results_dir, model: str, all_theta, all_theta_u, report_df, mle_theta, mle_u, acc_rates):
    os.makedirs(results_dir, exist_ok=True)
    M, N_post, _ = all_theta.shape
    combined = all_theta.reshape(-1, 4)
    combined_u = all_theta_u.reshape(-1, 4)
    chain_ids = np.repeat(np.arange(1, M + 1), N_post)

    df_post = pd.DataFrame(combined, columns=PARAM_NAMES)
    df_post.insert(0, "sample_id", range(len(combined)))
    df_post.insert(1, "chain_id", chain_ids)
    p_post = posterior_samples_path(results_dir, model)
    df_post.to_csv(p_post, index=False)

    df_post_u = pd.DataFrame(combined_u, columns=PARAM_NAMES_U)
    df_post_u.insert(0, "sample_id", range(len(combined_u)))
    df_post_u.insert(1, "chain_id", chain_ids)
    p_post_u = os.path.join(results_dir, f"posterior_samples_u_{model}_SV.csv")
    df_post_u.to_csv(p_post_u, index=False)

    p_diag = os.path.join(results_dir, f"diagnostics_{model}_SV.csv")
    report_df.to_csv(p_diag)

    df_mle = pd.DataFrame({
        "parameter": PARAM_NAMES,
        "mle_value": mle_theta,
        "theta_u_value": mle_u,
    })
    p_mle = os.path.join(results_dir, f"mle_{model}_SV.csv")
    df_mle.to_csv(p_mle, index=False)

    p_acc = os.path.join(results_dir, f"acceptance_rates_{model}_SV.csv")
    pd.DataFrame({"chain_id": range(1, M + 1), "acceptance_rate": acc_rates}).to_csv(p_acc, index=False)

    for p in (p_post, p_post_u, p_diag, p_mle, p_acc):
        print(f"[EXPORT] {p}")

    return {"posterior": p_post, "posterior_u": p_post_u, "diagnostics": p_diag, "mle": p_mle, "acceptance": p_acc}


def export_filtered_states(results_dir, model: str, df_raw, h_filt, P_filt, nu, P_rr_arr):
    df_out = df_raw[["timestamp", "close", "log_ret"]].copy()
    df_out["h_filtered"] = h_filt
    df_out["P_filtered"] = P_filt
    df_out["sigma_filtered"] = np.exp(h_filt / 2.0 + P_filt / 8.0)
    df_out["innovation"] = nu
    df_out["P_obs"] = P_rr_arr

    path = os.path.join(results_dir, f"filtered_states_{model}_SV.csv")
    df_out.to_csv(path, index=False)
    print(f"[EXPORT] {path}")
    return path
