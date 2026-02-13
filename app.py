import os
import random
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

from src.models.lstm_model import LSTMVolatility
from src.models.model_io import load_model
from src.pipeline.train_pipeline import TrainingPipeline
from src.risk_engine.risk_score import RiskScorer
from src.models.garch_model import GARCHModel
from src.data_loader.data_ingestion import MarketDataLoader


# ==========================
# Deterministic Behavior
# ==========================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


st.set_page_config(page_title="FinRisk-Engine", layout="wide")

st.title("📊 FinRisk-Engine — Institutional Volatility Intelligence")


# ==========================
# Sidebar Config
# ==========================
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.text_input("Start Date", "2020-01-01")
train_model = st.sidebar.button("Train Model")
run_forecast = st.sidebar.button("Run Forecast")


# ==========================
# TRAIN MODEL (Offline)
# ==========================
if train_model:

    st.info("Training volatility models...")

    pipeline = TrainingPipeline(
        tickers=[ticker],
        start_date=start_date
    )

    results = pipeline.run()

    st.success("Model training completed.")
    st.write(results)


# ==========================
# RUN FORECAST
# ==========================
if run_forecast:

    # Load historical data
    loader = MarketDataLoader([ticker], start_date)
    returns = loader.run().squeeze()

    # Current price
    price_data = yf.download(ticker, period="5d", progress=False)
    current_price = float(price_data["Close"].iloc[-1])

    # Volatility series
    window = 20
    vol_series = returns.rolling(window).std().dropna()

    # Load trained LSTM model
    model = LSTMVolatility(input_size=1)
    model = load_model(model, "models_artifacts/lstm_vol_model.pth")
    model.eval()

    latest_window = vol_series.values[-window:]
    x = torch.tensor(latest_window.reshape(1, window, 1), dtype=torch.float32)

    with torch.no_grad():
        predicted_vol = float(model(x).item())

    # GARCH forecast
    garch_model = GARCHModel()
    garch_model.fit(returns)
    garch_vol = float(garch_model.forecast(horizon=1).values[0])

    # Risk score
    drawdown = (returns.cumsum().min())
    risk_score = RiskScorer.compute(predicted_vol, drawdown)

    # Direction classification
    recent_return = returns.iloc[-1]

    if recent_return > 0:
        direction = "Bullish 📈"
        color = "green"
    else:
        direction = "Bearish 📉"
        color = "red"

    # ==========================
    # DISPLAY
    # ==========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Predicted Volatility (LSTM)", f"{predicted_vol:.4f}")
    col3.metric("GARCH Volatility", f"{garch_vol:.4f}")

    st.markdown(
        f"## Market Direction: <span style='color:{color}'>{direction}</span>",
        unsafe_allow_html=True
    )

    # Risk Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Composite Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "green"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"},
            ],
        },
    ))

    st.plotly_chart(fig, use_container_width=True)
