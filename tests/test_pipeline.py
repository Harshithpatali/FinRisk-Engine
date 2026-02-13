from src.pipeline.train_pipeline import TrainingPipeline


def test_training_pipeline():

    pipeline = TrainingPipeline(
        tickers=["AAPL"],
        start_date="2022-01-01"
    )

    results = pipeline.run()

    # Check keys exist
    assert "garch_vol_forecast" in results
    assert "lstm_loss" in results
    assert "sharpe" in results

    # Check types
    assert isinstance(results["garch_vol_forecast"], float)
    assert isinstance(results["lstm_loss"], float)
    assert isinstance(results["sharpe"], float)
