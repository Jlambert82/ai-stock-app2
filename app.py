import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Stock Predictor", layout="centered")

st.title("📈 AI Stock Predictor")
st.subheader("Short-Term (Max 1 Week Hold)")

# -----------------------------
# Get Stock Data
# -----------------------------
def get_stock_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")

    # 🚨 Handle invalid ticker
    if df.empty:
        return None

    # 🚨 Fix dataframe shape issues
    df = df[['Close', 'Volume']].copy()
    df['Close'] = df['Close'].squeeze()

    # Indicators
    df['rsi'] = RSIIndicator(close=df['Close']).rsi()
    macd = MACD(close=df['Close'])
    df['macd'] = macd.macd()

    # Target (next day up or down)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna()

# -----------------------------
# Train Model
# -----------------------------
def train_model(df):
    X = df[['Close', 'Volume', 'rsi', 'macd']]
    y = df['target']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model

# -----------------------------
# Predict
# -----------------------------
def predict(model, df):
    latest = df[['Close', 'Volume', 'rsi', 'macd']].iloc[-1:]
    prob = model.predict_proba(latest)[0][1]
    return prob

# -----------------------------
# Strategy
# -----------------------------
def get_strategy(price):
    return {
        "Take Profit (+5%)": round(price * 1.05, 2),
        "Stop Loss (-2%)": round(price * 0.98, 2),
        "Max Hold Days": 5
    }

# -----------------------------
# UI Input
# -----------------------------
ticker = st.text_input("Enter Stock Ticker", "AAPL")

# -----------------------------
# Run Analysis
# -----------------------------
if st.button("Analyze"):

    with st.spinner("Analyzing stock..."):
        df = get_stock_data(ticker)

        if df is None:
            st.error("❌ Invalid ticker or no data found.")
        else:
            model = train_model(df)
            prob = predict(model, df)
            price = df['Close'].iloc[-1]
            strategy = get_strategy(price)

            st.success(f"✅ Analysis for {ticker.upper()}")

            # Display results
            st.metric("💰 Current Price", f"${price:.2f}")
            st.metric("📊 Chance of Going Up", f"{prob:.2%}")

            st.subheader("📌 Trading Strategy")
            st.write(f"🎯 Take Profit: ${strategy['Take Profit (+5%)']}")
            st.write(f"🛑 Stop Loss: ${strategy['Stop Loss (-2%)']}")
            st.write(f"⏳ Max Hold: {strategy['Max Hold Days']} days")

            # Optional chart
            st.subheader("📉 Price Chart")
            st.line_chart(df['Close'])
