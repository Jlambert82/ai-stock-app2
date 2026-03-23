import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Stock Scanner", layout="wide")

st.title("🚀 AI Stock Scanner")
st.subheader("Short-Term AI Predictions (1–5 Days)")

stocks = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN",
    "META", "GOOGL", "AMD", "NFLX", "INTC"
]

# -----------------------------
# Color Function
# -----------------------------
def get_color(prob):
    if prob > 0.7:
        return "green"
    elif prob > 0.55:
        return "lightgreen"
    elif prob > 0.45:
        return "orange"
    else:
        return "red"

# -----------------------------
# Company Name
# -----------------------------
def get_company_name(ticker):
    try:
        return yf.Ticker(ticker).info.get("longName", ticker)
    except:
        return ticker

# -----------------------------
# Get Data
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

    # Indicators
    df_clean['rsi'] = RSIIndicator(close=df_clean['Close']).rsi()
    macd = MACD(close=df_clean['Close'])
    df_clean['macd'] = macd.macd()

    # Targets
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
# UI
# -----------------------------
if st.button("🔍 Scan Market"):

    cols = st.columns(3)

    for i, ticker in enumerate(stocks):
        try:
            df = get_stock_data(ticker)

            if df is None:
                st.write(f"{ticker} - No data")
                continue

            models = train_models(df)
            probs = predict(models, df)

            price = df['Close'].iloc[-1]
            company = get_company_name(ticker)

            with cols[i % 3]:
                st.markdown(f"### 📊 {company}")
                st.caption(ticker)

                st.metric("💰 Price", f"${price:.2f}")

                # Probabilities
                for label, prob in probs.items():
                    color = get_color(prob)
                    st.markdown(
                        f"<span style='color:{color}; font-size:18px;'>📅 {label}: {prob:.2%}</span>",
                        unsafe_allow_html=True
                    )

                # Signal
                avg_prob = sum(probs.values()) / 3
                if avg_prob > 0.65:
                    signal = "🟢 Good Chance to Go Up"
                elif avg_prob > 0.5:
                    signal = "🟡 Could Go Up"
                else:
                    signal = "🔴 Low Chance"

                st.write(f"🧠 {signal}")

                # Chart
                st.line_chart(df["Close"])

                # Explanation
                with st.expander("ℹ️ What this means (simple)"):
                    rsi = df['rsi'].iloc[-1]
                    macd = df['macd'].iloc[-1]

                    if rsi > 70:
                        rsi_text = "The stock went up a lot recently — it might slow down or drop soon."
                    elif rsi < 30:
                        rsi_text = "The stock dropped a lot recently — it might bounce back up."
                    else:
                        rsi_text = "The stock is moving normally — nothing extreme."

                    if macd > 0:
                        macd_text = "The trend is currently going UP."
                    else:
                        macd_text = "The trend is currently going DOWN."

                    st.write("🧠 **How the AI is thinking:**")
                    st.write("This AI studies price trends and activity to estimate if a stock might go up.")

                    st.write("---")

                    st.write(f"📊 Momentum (RSI: {rsi:.2f})")
                    st.write(rsi_text)

                    st.write("---")

                    st.write(f"📈 Trend (MACD: {macd:.2f})")
                    st.write(macd_text)

                    st.write("---")

                    st.write("📅 Timeframes:")
                    st.write("• 1 Day → Tomorrow")
                    st.write("• 3 Day → Short-term")
                    st.write("• 5 Day → About a week")

                    st.write("---")

                    st.write("🎯 Colors:")
                    st.write("🟢 High chance up")
                    st.write("🟡 Medium")
                    st.write("🔴 Low chance")

                st.divider()

        except Exception as e:
            st.write(f"{ticker} error")
