import os
import numpy as np
import pandas as pd
import yfinance as yf

def download_dataset(ticker: str = "SPY", 
                     start_date: str = "2018-03-13", 
                     end_date: str = "2026-05-30", 
                     output_dir: str = "dataset", 
                     vol_lower_sigma: float = 2.0) -> str:
    """
    Descarga datos de Yahoo Finance, filtra por días laborables y volumen,
    muestra las estadísticas descriptivas y guarda en CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dataset_{ticker}_daily.csv")
    
    print(f"[DOWNLOAD] Descargando {ticker} desde {start_date} hasta {end_date}...")
    raw = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw[["Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df.reset_index().rename(columns={"Close": "close", "Volume": "volume"})

    # 1. Filtro días entre semana
    df = df[df["timestamp"].dt.dayofweek < 5].copy()

    # 2. Filtro por volumen
    mu_vol = df["volume"].mean()
    sigma_vol = df["volume"].std(ddof=1)
    vol_floor = mu_vol - vol_lower_sigma * sigma_vol

    mask_vol = df["volume"] >= vol_floor
    n_removed = (~mask_vol).sum()
    df = df[mask_vol].copy()

    print(
        f"Filtro de volumen: umbral = {vol_floor:,.0f} acciones "
        f"(mu={mu_vol:,.0f}, sigma={sigma_vol:,.0f})\n"
        f"Sesiones eliminadas: {n_removed}"
    )

    # 3. Log-retornos
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_ret"]).reset_index(drop=True)

    # 4. Estadísticos descriptivos
    T = len(df)
    r = df["log_ret"]
    print(f"\nObservaciones (T):       {T}")
    print(f"Fecha inicio:            {df['timestamp'].iloc[0].date()}")
    print(f"Fecha fin:               {df['timestamp'].iloc[-1].date()}")
    print(f"Media diaria:            {r.mean():.6f}")
    print(f"Desv. típica diaria:     {r.std():.6f}")
    print(f"Volatilidad anualizada:  {r.std() * np.sqrt(252) * 100:.2f}%")
    print(f"Asimetría:               {r.skew():.4f}")
    print(f"Curtosis en exceso:      {r.kurtosis():.4f}")
    print(f"Mínimo:                  {r.min():.6f}")
    print(f"Máximo:                  {r.max():.6f}")

    # 5. Guardado
    df.to_csv(output_path, index=False)
    print(f"\n[DATA] Guardado exitosamente en: {output_path}")
    return output_path


def load_data(filepath: str, start_date: str = None, end_date: str = None):
    """
    Carga el dataset CSV y permite acotar por rango de fechas dinámicamente.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if start_date:
        df = df[df["timestamp"] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df["timestamp"] <= pd.to_datetime(end_date, utc=True)]

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_ret"]).reset_index(drop=True)

    print(f"[DATA] N={len(df)} retornos  |  {df.timestamp.iloc[0].date()} → {df.timestamp.iloc[-1].date()}")
    print(f"[DATA] Retorno medio diario: {df.log_ret.mean():.6f}")
    print(f"[DATA] Std diaria:           {df.log_ret.std():.6f}  ({df.log_ret.std()*np.sqrt(252)*100:.2f}% anualizada)")
    return df["log_ret"].values, df
