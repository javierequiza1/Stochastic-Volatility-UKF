"""Orquestador del pipeline SV-AUKF para un modelo dado ('L2' o 'L3').

Logica de cache acordada: si ya existe posterior_samples_{model}_SV.csv en
el directorio de resultados, se reutiliza tal cual (no se recalcula MLE ni
MCMC); si no existe (o force_recompute=True), se estima desde cero (MLE
multistart -> MCMC -> diagnosticos -> export) y luego, en ambos casos, se
corre el filtro final con la media posterior para obtener y exportar los
estados filtrados y los diagnosticos de innovaciones.
"""
from __future__ import annotations

import os

from .config import MCMC_CONFIG, PRIORS_DIR, results_dir_for
from .data_loader import load_data
from .diagnostics import residual_diagnostics, summarize_mcmc
from .filters import RUN_AUKF
from .io_utils import (
    export_filtered_states,
    export_mcmc_outputs,
    has_cached_posterior,
    load_cached_posterior,
)
from .mcmc import run_mcmc
from .mle import estimate_mle
from .priors import load_priors
from .transforms import theta_to_unconstrained


def run_model(
    model: str,
    dataset_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
    force_recompute: bool = False,
    verbose: bool = True,
):
    """model: 'L2' o 'L3'."""
    assert model in ("L2", "L3"), "model debe ser 'L2' o 'L3'"

    results_dir = results_dir_for(model)
    priors_path = os.path.join(PRIORS_DIR, f"priors_{model}.json")
    priors = load_priors(priors_path)

    returns, df_raw = load_data(dataset_path, start_date=start_date, end_date=end_date)

    if not force_recompute and has_cached_posterior(results_dir, model):
        print(f"[CACHE] Parametros ya optimizados encontrados para {model} en {results_dir} -- se reutilizan.")
        combined = load_cached_posterior(results_dir, model)
        post_mean_theta = combined.mean(axis=0)
    else:
        print(f"[CACHE] No hay resultados previos para {model} (o force_recompute=True) -- estimando desde cero.")
        mle_theta, mle_ll = estimate_mle(returns, model, priors["mle"], verbose=verbose)
        all_theta, all_theta_u, acc_rates = run_mcmc(
            returns, model, mle_theta, priors["bayes"], cfg=MCMC_CONFIG,
        )
        report = summarize_mcmc(all_theta)
        print(f"\n{report.round(4).to_string()}")

        export_mcmc_outputs(
            results_dir, model, all_theta, all_theta_u, report,
            mle_theta, theta_to_unconstrained(mle_theta), acc_rates,
        )
        post_mean_theta = report["Post Mean"].values

    # Filtro final con la media posterior -> estados filtrados + diagnostico de innovaciones
    ll_f, h_filt, P_filt, nu, P_rr_arr = RUN_AUKF[model](returns, tuple(post_mean_theta), return_states=True)
    if ll_f <= -1e10 or h_filt is None:
        raise RuntimeError(f"[{model}] El filtro diverge en la media posterior -- revisar parametros.")

    export_filtered_states(results_dir, model, df_raw, h_filt, P_filt, nu, P_rr_arr)
    residual_diagnostics(
        nu, P_rr_arr,
        output_path=os.path.join(results_dir, f"innovation_diagnostics_{model}.txt"),
    )

    return {
        "post_mean_theta": post_mean_theta,
        "h_filtered": h_filt,
        "P_filtered": P_filt,
        "innovation": nu,
        "P_obs": P_rr_arr,
        "results_dir": str(results_dir),
    }
