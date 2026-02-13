import random
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from datetime import date

from src.models.lstm_model import LSTMVolatility
from src.models.model_io import load_model
from src.pipeline.train_pipeline import TrainingPipeline
from src.risk_engine.risk_score import RiskScorer
from src.models.garch_model import GARCHModel
from src.data_loader.data_ingestion import MarketDataLoader
from src.models.hmm_model import RegimeHMM


# ==========================
# Deterministic Behavior
# ==========================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="FinRisk-Engine", layout="wide")

st.title("📊 FinRisk-Engine — Institutional Risk Terminal")


# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("Market Configuration")

popular_stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "SPY", "QQQ"
]

ticker_list = st.sidebar.multiselect(
    "Select Portfolio Stocks",
    options=popular_stocks,
    default=["AAPL"]
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2020, 1, 1)
)

run_analysis = st.sidebar.button("Run Risk Analysis")


# ==========================
# ANALYSIS
# ==========================
if run_analysis:

    portfolio_data = {}
    portfolio_returns = []

    for ticker in ticker_list:

        loader = MarketDataLoader([ticker], str(start_date))
        returns = loader.run().squeeze()
        portfolio_returns.append(returns)

        price_data = yf.download(ticker, start=str(start_date), progress=False)

        # ----------------------------
        # LIVE PRICE CHART
        # ----------------------------
        st.markdown(f"## 📈 {ticker} Price Chart")

        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data["Close"],
                name="Close Price"
            )
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # ----------------------------
        # VOLATILITY HISTORY
        # ----------------------------
        window = 20
        vol_series = returns.rolling(window).std().dropna()

        st.markdown("### 📊 Rolling Volatility")

        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=vol_series.index,
                y=vol_series.values,
                name="Rolling Volatility"
            )
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # ----------------------------
        # DRAWdown
        # ----------------------------
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak

        st.markdown("### 📉 Rolling Drawdown")

        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown"
            )
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ----------------------------
        # VaR & CVaR
        # ----------------------------
        confidence_level = 0.95
        var = np.percentile(returns, 100 * (1 - confidence_level))
        cvar = returns[returns <= var].mean()

        st.markdown("### 🛑 Risk Metrics")

        col1, col2 = st.columns(2)
        col1.metric("VaR (95%)", f"{var:.4f}")
        col2.metric("CVaR (95%)", f"{cvar:.4f}")

        # ----------------------------
        # REGIME DETECTION
        # ----------------------------
        st.markdown("### 🔍 Market Regime")

        hmm = RegimeHMM(n_states=2)
        hmm.fit(returns.values)
        states, _ = hmm.predict(returns.values)

        fig_regime = go.Figure()
        fig_regime.add_trace(
            go.Scatter(
                x=returns.index,
                y=states,
                mode="markers",
                name="Regime State"
            )
        )
        st.plotly_chart(fig_regime, use_container_width=True)

        # Store for portfolio aggregation
        portfolio_data[ticker] = {
            "returns": returns,
            "volatility": vol_series,
            "drawdown": drawdown,
            "VaR": var,
            "CVaR": cvar
        }

    # ==========================
    # PORTFOLIO VIEW
    # ==========================
    if len(ticker_list) > 1:

        st.markdown("## 🏦 Portfolio Risk Overview")

        combined_returns = pd.concat(portfolio_returns, axis=1).mean(axis=1)

        portfolio_vol = combined_returns.rolling(20).std().iloc[-1]
        portfolio_var = np.percentile(combined_returns, 5)
        portfolio_cvar = combined_returns[combined_returns <= portfolio_var].mean()

        risk_score = RiskScorer.compute(
            portfolio_vol,
            combined_returns.cumsum().min()
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Portfolio Volatility", f"{portfolio_vol:.4f}")
        col2.metric("Portfolio VaR (95%)", f"{portfolio_var:.4f}")
        col3.metric("Portfolio CVaR (95%)", f"{portfolio_cvar:.4f}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Portfolio Risk Score"},
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

    # ==========================
    # EXPLANATION PANEL
    # ==========================
    with st.expander("ℹ️ Explanation of Metrics"):

        st.markdown("""
### Live Price Chart
Displays historical closing prices.

### Rolling Volatility
20-day rolling standard deviation of returns.
Measures short-term market uncertainty.

### Rolling Drawdown
Peak-to-trough decline from cumulative returns.

### VaR (Value at Risk)
Maximum expected loss at 95% confidence.

### CVaR (Conditional VaR)
Average loss beyond VaR threshold.
Measures tail risk severity.

### Regime Detection
Hidden Markov Model classification of market states
(e.g., low-volatility vs high-volatility regime).

### Portfolio Risk Score
Composite measure of:
- Portfolio volatility
- Historical drawdown
        """)
