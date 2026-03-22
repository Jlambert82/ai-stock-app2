import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Stock Scanner", layout="wide")

st.title("🚀 AI Stock Scanner")
st.subheader("Find Top Short-Term Trades (Max 1 Week Hold)")

# -----------------------------
# STOCK LIST (you can expand this)
# -----------------------------
stocks = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN",
    "META", "GOOGL", "AMD", "NFLX", "INTC",
    "SPY", "QQQ", "COIN", "PLTR", "BA"
]

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
# SCAN BUTTON
# -----------------------------
if st.button("🔍 Scan Market"):

    results = []

    progress = st.progress(0)

    for i, ticker in enumerate(stocks):
        try:
            df = get_stock_data(ticker)

            if df is None:
                continue

            model = train_model(df)
            prob = predict(model, df)
            price = df['Close'].iloc[-1]

            strategy = get_strategy(price)

            results.append({
                "Ticker": ticker,
                "Price": round(price, 2),
                "Probability": round(prob * 100, 2),
                "Take Profit": strategy["Take Profit"],
                "Stop Loss": strategy["Stop Loss"]
            })

        except:
            continue

        progress.progress((i + 1) / len(stocks))

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by best probability
    results_df = results_df.sort_values(by="Probability", ascending=False)

    # Show Top 5
    st.subheader("🏆 Top 5 Stocks Right Now")
    st.dataframe(results_df.head(5), use_container_width=True)

    # Show full table
    st.subheader("📊 Full Scan Results")
    st.dataframe(results_df, use_container_width=True)
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
