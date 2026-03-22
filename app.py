import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Stock Scanner", layout="wide")

st.title("🚀 AI Stock Scanner")
st.subheader("Top Short-Term Opportunities (Max 1 Week Hold)")

# -----------------------------
# STOCK LIST
# -----------------------------
stocks = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN",
    "META", "GOOGL", "AMD", "NFLX", "INTC"
]

# -----------------------------
# GET COMPANY NAME
# -----------------------------
def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName", ticker)
    except:
        return ticker

# -----------------------------
# DATA FUNCTION (FIXED)
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

    df_clean['target'] = (df_clean['Close'].shift(-1) > df_clean['Close']).astype(int)

    return df_clean.dropna()

# -----------------------------
# MODEL
# -----------------------------
def train_model(df):
    X = df[['Close', 'Volume', 'rsi', 'macd']]
    y = df['target']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model

# -----------------------------
# PREDICT
# -----------------------------
def predict(model, df):
    latest = df[['Close', 'Volume', 'rsi', 'macd']].iloc[-1:]
    prob = model.predict_proba(latest)[0][1]
    return prob

# -----------------------------
# STRATEGY
# -----------------------------
def get_strategy(price):
    return {
        "Take Profit": round(price * 1.05, 2),
        "Stop Loss": round(price * 0.98, 2),
        "Max Hold": 5
    }

# -----------------------------
# SCAN
# -----------------------------
if st.button("🔍 Scan Market"):

    cols = st.columns(3)  # 3 cards per row

    for i, ticker in enumerate(stocks):
        try:
            df = get_stock_data(ticker)
            if df is None:
                continue

            model = train_model(df)
            prob = predict(model, df)
            price = df['Close'].iloc[-1]
            strategy = get_strategy(price)
            company_name = get_company_name(ticker)

            with cols[i % 3]:
                st.markdown("### 📊 " + company_name)
                st.caption(ticker)

                # Metrics
                st.metric("Price", f"${price:.2f}")
                st.metric("Chance ↑", f"{prob:.2%}")

                # Strategy
                st.write("🎯 TP:", strategy["Take Profit"])
                st.write("🛑 SL:", strategy["Stop Loss"])
                st.write("⏳ Hold:", str(strategy["Max Hold"]) + " days")

                # Chart
                st.line_chart(df["Close"])

                st.divider()

        except Exception as e:
            st.write(f"{ticker} error")
