from data_loader import download_dataset

if __name__ == "__main__":
    # Puedes modificar el ticker o fechas aquí según tus requerimientos
    TICKER = "SPY"
    START_DATE = "2018-03-13"
    END_DATE = "2026-05-30"

    download_dataset(ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
