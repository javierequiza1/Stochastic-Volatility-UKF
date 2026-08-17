"""Descarga el dataset de mercado y lo guarda en dataset/dataset_{TICKER}_daily.csv.

Uso:
    python download_data.py
"""
from sv_aukf.config import DEFAULT_END, DEFAULT_START, DEFAULT_TICKER
from sv_aukf.data_loader import download_dataset

if __name__ == "__main__":
    download_dataset(ticker=DEFAULT_TICKER, start_date=DEFAULT_START, end_date=DEFAULT_END)
