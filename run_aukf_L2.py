import os
import pandas as pd
import numpy as np
from src.data_loader import load_data
from src.models import estimate_mle_L2, run_aukf_L2
from src.utils import load_priors, diagnostics

def main():
    # Rutas relativas limpia
    data_path = os.path.join("dataset", "dataset_SPY_daily.csv")
    priors_path = os.path.join("priors", "priors_L2.json")
    out_path = os.path.join("dataset", "aukf_sv_results_L2.csv")

    priors = load_priors(priors_path)
    returns, df_raw = load_data(data_path, start_date="2018-03-13", end_date="2026-05-30")

    print("\n[ESTIMACIÓN] Iniciando MLE para SV-AUKF (L=2)...")
    alpha, phi, sigma_eta, rho, mu1, log_lik = estimate_mle_L2(returns, priors)

    print(f"\nResultados MLE L=2: alpha={alpha:.4f}, phi={phi:.4f}, sigma={sigma_eta:.4f}, rho={rho:.4f}, loglik={log_lik:.4f}")

    _, h_filt, P_filt, nu, P_rr_arr = run_aukf_L2(returns, (alpha, phi, sigma_eta, rho), return_states=True)

    diagnostics(nu, P_rr_arr)

    df_out = df_raw[["timestamp", "close", "log_ret"]].copy()
    df_out["h_filtered"] = h_filt
    df_out["sigma_filtered"] = np.exp(h_filt / 2.0 + P_filt / 8.0)
    df_out["innovation"] = nu

    df_out.to_csv(out_path, index=False)
    print(f"\n[OUT] Guardado resultados L=2 en: {out_path}")

if __name__ == "__main__":
    main()
