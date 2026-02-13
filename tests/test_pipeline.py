from src.pipeline.train_pipeline import TrainingPipeline


def test_training_pipeline():

    pipeline = TrainingPipeline(
        tickers=["AAPL"],
        start_date="2022-01-01"
    )

    results = pipeline.run()

    assert "garch_forecast" in results
    assert "lstm_loss" in results
    assert "sharpe" in results
