import os
import json
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, chi2

def load_priors(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encuentra el archivo de priors: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def diagnostics(nu: np.ndarray, P_rr_arr: np.ndarray, output_path: str = None):
    """Calcula diagnósticos, los imprime en consola y opcionalmente los guarda en archivo."""
    nu_std = nu / np.sqrt(P_rr_arr)

    jb_stat_raw, jb_p_raw = jarque_bera(nu)
    jb_stat_std, jb_p_std = jarque_bera(nu_std)

    def ljung_box(x, lags):
        n = len(x)
        x_c = x - x.mean()
        c0 = np.dot(x_c, x_c) / n
        acf = [np.dot(x_c[:-k], x_c[k:]) / (n * c0) for k in range(1, lags + 1)]
        Q = n * (n + 2) * sum(acf[k-1]**2 / (n - k) for k in range(1, lags + 1))
        p_val = 1 - chi2.cdf(Q, df=lags)
        return Q, p_val

    nu2 = nu / nu.std()

    lb10_nu,  p10_nu  = ljung_box(nu2,      10)
    lb20_nu,  p20_nu  = ljung_box(nu2,      20)
    lb10_nu2, p10_nu2 = ljung_box(nu2**2,   10)
    lb20_nu2, p20_nu2 = ljung_box(nu2**2,   20)
    lb10_std, p10_std = ljung_box(nu_std,  10)
    lb10_s2,  p10_s2  = ljung_box(nu_std**2, 10)

    corr_nu  = np.corrcoef(nu[:-1],     nu[1:]    )[0, 1]
    corr_std = np.corrcoef(nu_std[:-1], nu_std[1:])[0, 1]

    lines = [
        "="*62,
        "DIAGNÓSTICOS DE INNOVACIONES CRUDAS  (nu_t)",
        "="*62,
        f"  Media:                   {nu2.mean():.6f}   (ideal: 0)",
        f"  Std:                     {nu2.std():.6f}   (ideal: 1)",
        f"  Sesgo:                   {pd.Series(nu2).skew():.4f}   (ideal: 0)",
        f"  Curtosis exceso:         {pd.Series(nu2).kurt():.4f}   (ideal: 0)",
        f"  Jarque-Bera:             stat={jb_stat_raw:.2f}   p={jb_p_raw:.4f}  {'✓' if jb_p_raw > 0.05 else '✗'}",
        f"  Ljung-Box(nu,  lag=10):  Q={lb10_nu:.2f}   p={p10_nu:.4f}",
        f"  Ljung-Box(nu,  lag=20):  Q={lb20_nu:.2f}   p={p20_nu:.4f}",
        f"  Ljung-Box(nu², lag=10):  Q={lb10_nu2:.2f}   p={p10_nu2:.4f}",
        f"  Ljung-Box(nu², lag=20):  Q={lb20_nu2:.2f}   p={p20_nu2:.4f}",
        f"  ACF(nu, lag=1):          {corr_nu:.4f}   (ideal: 0)",
        "",
        "="*62,
        "DIAGNÓSTICOS DE INNOVACIONES ESTANDARIZADAS  (e_t = nu_t / sqrt(P_rr_t))",
        "="*62,
        f"  Media:                   {nu_std.mean():.6f}   (ideal: 0)",
        f"  Std:                     {nu_std.std():.6f}   (ideal: 1)",
        f"  Sesgo:                   {pd.Series(nu_std).skew():.4f}   (ideal: 0)",
        f"  Curtosis exceso:         {pd.Series(nu_std).kurt():.4f}   (ideal: 0)",
        f"  Jarque-Bera:             stat={jb_stat_std:.2f}   p={jb_p_std:.4f}  {'✓' if jb_p_std > 0.05 else '✗'}",
        f"  Ljung-Box(e,  lag=10):   Q={lb10_std:.2f}   p={p10_std:.4f}",
        f"  Ljung-Box(e², lag=10):   Q={lb10_s2:.2f}   p={p10_s2:.4f}",
        f"  ACF(e, lag=1):           {corr_std:.4f}   (ideal: 0)",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[OUT] Diagnósticos guardados en:\n      {output_path}")
