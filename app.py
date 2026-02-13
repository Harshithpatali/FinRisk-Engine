import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import yfinance as yf

from src.pipeline.train_pipeline import TrainingPipeline
from src.models.lstm_model import LSTMVolatility
from src.models.model_io import load_model
from src.models.uncertainty import mc_dropout_predict
from src.models.hmm_model import RegimeHMM
from src.risk_engine.risk_score import RiskScorer


st.set_page_config(page_title="FinRisk-Engine", layout="wide")

st.title("📊 FinRisk-Engine — Institutional Market Intelligence")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.text_input("Start Date", "2020-01-01")
run_button = st.sidebar.button("Run Analysis")

if run_button:

    # ===============================
    # 1️⃣ Fetch Current Price
    # ===============================
    data = yf.download(ticker, period="5d", progress=False)
    current_price = float(data["Close"].iloc[-1])
    previous_close = float(data["Close"].iloc[-2])

    # ===============================
    # 2️⃣ Run Training Pipeline
    # ===============================
    pipeline = TrainingPipeline(
        tickers=[ticker],
        start_date=start_date
    )

    results = pipeline.run()

    # ===============================
    # 3️⃣ Load LSTM Model
    # ===============================
    model = LSTMVolatility()
    model = load_model(model, "models_artifacts/lstm_model.pth")

    x = torch.randn(1, 20, 1)

    mean_pred, std_pred = mc_dropout_predict(model, x)

    predicted_return = float(mean_pred.item())
    uncertainty = float(std_pred.item())

    # ===============================
    # 4️⃣ Direction & % Move
    # ===============================
    predicted_price = current_price * (1 + predicted_return)
    percent_move = predicted_return * 100

    if predicted_return > 0:
        direction = "Bullish 📈"
        color = "green"
    else:
        direction = "Bearish 📉"
        color = "red"

    # ===============================
    # 5️⃣ Confidence Score
    # ===============================
    confidence = max(0, 100 - (uncertainty * 1000))
    confidence = min(confidence, 100)

    # ===============================
    # 6️⃣ Regime Detection
    # ===============================
    returns = np.random.randn(300)
    hmm = RegimeHMM(n_states=2)
    hmm.fit(returns)
    states, probs = hmm.predict(returns)
    latest_regime_prob = probs[-1]

    # ===============================
    # 7️⃣ Risk Score
    # ===============================
    risk_score = RiskScorer.compute(
        volatility=abs(predicted_return),
        drawdown=-0.2
    )

    # ===============================
    # DASHBOARD DISPLAY
    # ===============================

    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Predicted Move", f"{percent_move:.2f}%")
    col3.metric("Model Confidence", f"{confidence:.1f}%")

    st.markdown(f"## Market Direction: <span style='color:{color}'>{direction}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Risk Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Composite Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"},
            "steps": [
                {"range": [0, 30], "color": "green"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"},
            ],
        },
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Regime Probabilities
    st.subheader("Regime Probabilities")

    fig_regime = go.Figure()
    for i in range(len(latest_regime_prob)):
        fig_regime.add_trace(
            go.Bar(
                x=[f"Regime {i}"],
                y=[latest_regime_prob[i]]
            )
        )

    st.plotly_chart(fig_regime, use_container_width=True)

    # Forecast Band
    st.subheader("Forecast with Uncertainty Band")

    x_axis = np.arange(10)
    vol_forecast = np.full(10, predicted_price)

    upper = vol_forecast * (1 + uncertainty)
    lower = vol_forecast * (1 - uncertainty)

    fig_band = go.Figure()
    fig_band.add_trace(go.Scatter(x=x_axis, y=vol_forecast, name="Predicted Price"))
    fig_band.add_trace(go.Scatter(x=x_axis, y=upper, name="Upper Band"))
    fig_band.add_trace(go.Scatter(x=x_axis, y=lower, name="Lower Band"))

    st.plotly_chart(fig_band, use_container_width=True)
