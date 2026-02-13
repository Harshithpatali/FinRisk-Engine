import random
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from datetime import date

from src.risk_engine.risk_score import RiskScorer
from src.models.hmm_model import RegimeHMM
from src.data_loader.data_ingestion import MarketDataLoader


# ==========================
# Deterministic Setup
# ==========================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="FinRisk-Engine", layout="wide")
st.title("📊 FinRisk-Engine — Institutional Risk Terminal")


# ==========================
# SIDEBAR CONFIG
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
# ANALYSIS ENGINE
# ==========================
if run_analysis:

    portfolio_returns = []

    for ticker in ticker_list:

        st.markdown(f"# 📌 {ticker} Analysis")

        loader = MarketDataLoader([ticker], str(start_date))
        returns = loader.run().squeeze()
        portfolio_returns.append(returns)

        price_data = yf.download(ticker, start=str(start_date), progress=False)
        current_price = float(price_data["Close"].iloc[-1])

        recent_return = returns.iloc[-1]
        direction = "Bullish 📈" if recent_return > 0 else "Bearish 📉"
        color = "green" if recent_return > 0 else "red"

        # ----------------------------
        # LIVE PRICE + SIGNAL
        # ----------------------------
        col1, col2 = st.columns(2)

        col1.metric("Current Price", f"${current_price:.2f}")
        col2.markdown(
            f"### Market Sentiment: <span style='color:{color}'>{direction}</span>",
            unsafe_allow_html=True
        )

        # ----------------------------
        # PRICE CHART
        # ----------------------------
        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data["Close"],
                name="Close Price"
            )
        )
        fig_price.update_layout(title="Price Chart")
        st.plotly_chart(fig_price, use_container_width=True)

        # ----------------------------
        # VOLATILITY
        # ----------------------------
        window = 20
        vol_series = returns.rolling(window).std().dropna()
        current_vol = vol_series.iloc[-1]

        # Normalize volatility score (scaled to 0–100)
        vol_score = min(current_vol * 1000, 100)

        st.markdown("### 📊 Volatility Analysis")

        col1, col2 = st.columns(2)
        col1.metric("Current Volatility", f"{current_vol:.4f}")
        col2.metric("Volatility Score (0–100)", f"{vol_score:.1f}")

        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=vol_series.index,
                y=vol_series.values,
                name="Rolling Volatility"
            )
        )
        fig_vol.update_layout(title="Rolling Volatility (20-Day)")
        st.plotly_chart(fig_vol, use_container_width=True)

        # ----------------------------
        # DRAWdown
        # ----------------------------
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        current_dd = drawdown.iloc[-1]

        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown"
            )
        )
        fig_dd.update_layout(title="Rolling Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

        # ----------------------------
        # VaR & CVaR
        # ----------------------------
        confidence_level = 0.95
        var = np.percentile(returns, 100 * (1 - confidence_level))
        cvar = returns[returns <= var].mean()

        st.markdown("### 🛑 Tail Risk Metrics")

        col1, col2 = st.columns(2)
        col1.metric("VaR (95%)", f"{var:.4f}")
        col2.metric("CVaR (95%)", f"{cvar:.4f}")

        # ----------------------------
        # REGIME DETECTION
        # ----------------------------
        st.markdown("### 🔍 Regime Detection")

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
        fig_regime.update_layout(title="Market Regime States")
        st.plotly_chart(fig_regime, use_container_width=True)

        # ----------------------------
        # RISK SCORE
        # ----------------------------
        risk_score = RiskScorer.compute(current_vol, current_dd)

        st.markdown("### ⚖ Composite Risk Score")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Risk Score (0–100)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"},
                ],
            },
        ))

        st.plotly_chart(fig_gauge, use_container_width=True)

        # ----------------------------
        # RISK PIE CHART
        # ----------------------------
        st.markdown("### 🧩 Risk Composition")

        risk_components = {
            "Volatility Risk": abs(current_vol),
            "Drawdown Risk": abs(current_dd),
            "Tail Risk (CVaR)": abs(cvar)
        }

        fig_pie = go.Figure(
            data=[go.Pie(
                labels=list(risk_components.keys()),
                values=list(risk_components.values()),
                hole=0.4
            )]
        )

        fig_pie.update_layout(title="Risk Contribution Breakdown")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

    # ==========================
    # PORTFOLIO VIEW
    # ==========================
    if len(ticker_list) > 1:

        st.markdown("# 🏦 Portfolio Overview")

        combined_returns = pd.concat(portfolio_returns, axis=1).mean(axis=1)

        portfolio_vol = combined_returns.rolling(20).std().iloc[-1]
        portfolio_dd = (combined_returns.cumsum().min())

        portfolio_risk_score = RiskScorer.compute(
            portfolio_vol,
            portfolio_dd
        )

        col1, col2 = st.columns(2)
        col1.metric("Portfolio Volatility", f"{portfolio_vol:.4f}")
        col2.metric("Portfolio Risk Score", f"{portfolio_risk_score:.1f}")


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
