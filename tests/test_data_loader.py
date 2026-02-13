import pytest
from src.data_loader.data_ingestion import MarketDataLoader


def test_data_download():
    loader = MarketDataLoader(
        tickers=["AAPL", "MSFT"],
        start_date="2022-01-01",
        end_date="2022-12-31",
    )

    df = loader.run()

    assert df is not None
    assert df.shape[0] > 0
    assert df.shape[1] == 2
