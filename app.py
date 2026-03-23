import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Stock Scanner", layout="wide")

st.title("🚀 AI Stock Scanner")
st.subheader("Short-Term Probability + Strategy Insights")

stocks = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN",
    "META", "GOOGL", "AMD", "NFLX", "INTC"
]

# -----------------------------
# Company Name
# -----------------------------
def get_company_name(ticker):
    try:
        return yf.Ticker(ticker).info.get("longName", ticker)
    except:
        return ticker

# -----------------------------
# Data
# -----------------------------
def get_stock_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")

    if df.empty:
        return None

    df = df.reset_index()

    close = pd.Series(df["Close"].values.flatten())
    volume = pd.Series(df["Volume"].values.flatten())

    df_clean = pd.DataFrame({
        "Close": close,
        "Volume": volume
    })

    df_clean['rsi'] = RSIIndicator(close=df_clean['Close']).rsi()
    macd = MACD(close=df_clean['Close'])
    df_clean['macd'] = macd.macd()

    # MULTI-TIME TARGETS
    df_clean['target_1d'] = (df_clean['Close'].shift(-1) > df_clean['Close']).astype(int)
    df_clean['target_3d'] = (df_clean['Close'].shift(-3) > df_clean['Close']).astype(int)
    df_clean['target_5d'] = (df_clean['Close'].shift(-5) > df_clean['Close']).astype(int)

    return df_clean.dropna()

# -----------------------------
# Train Models
# -----------------------------
def train_models(df):
    X = df[['Close', 'Volume', 'rsi', 'macd']]

    models = {}
    for label in ['target_1d', 'target_3d', 'target_5d']:
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, df[label])
        models[label] = model

    return models

# -----------------------------
# Predict
# -----------------------------
def predict(models, df):
    latest = df[['Close', 'Volume', 'rsi', 'macd']].iloc[-1:]

    return {
        "1 Day": models['target_1d'].predict_proba(latest)[0][1],
        "3 Day": models['target_3d'].predict_proba(latest)[0][1],
        "5 Day": models['target_5d'].predict_proba(latest)[0][1]
    }

# -----------------------------
# Strategy Logic
# -----------------------------
def get_strategy(price, prob_5d):
    if prob_5d > 0.7:
        return "🔥 Strong Buy"
    elif prob_5d > 0.55:
        return "👍 Moderate Buy"
    elif prob_5d > 0.45:
        return "⚠️ Risky"
    else:
        return "❌ Avoid"

# -----------------------------
# UI
# -----------------------------
if st.button("🔍 Scan Market"):

    cols = st.columns(3)

    for i, ticker in enumerate(stocks):
        try:
            df = get_stock_data(ticker)
            if df is None:
                continue

            models = train_models(df)
            probs = predict(models, df)

            price = df['Close'].iloc[-1]
            company = get_company_name(ticker)

            sentiment = get_strategy(price, probs["5 Day"])

            with cols[i % 3]:
                st.markdown(f"### 📊 {company}")
                st.caption(ticker)

                st.metric("💰 Price", f"${price:.2f}")

                # SHOW ALL PROBABILITIES (even bad ones)
                st.write(f"📅 1D: {probs['1 Day']:.2%}")
                st.write(f"📅 3D: {probs['3 Day']:.2%}")
                st.write(f"📅 5D: {probs['5 Day']:.2%}")

                st.write(f"🧠 Signal: {sentiment}")

                # Chart
                st.line_chart(df["Close"])

                # ℹ️ INFO DROPDOWN
                with st.expander("ℹ️ More Info"):
                    st.write("RSI:", round(df['rsi'].iloc[-1], 2))
                    st.write("MACD:", round(df['macd'].iloc[-1], 2))
                    st.write("Volume:", int(df['Volume'].iloc[-1]))

                    st.write("📘 Explanation:")
                    st.write("- RSI shows momentum")
                    st.write("- MACD shows trend direction")
                    st.write("- Probabilities are AI predictions")

                st.divider()

        except:
            st.write(f"{ticker} error")
