import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Stock Scanner", layout="wide")

# -----------------------------
# SESSION STATE (SAVE SELECTED STOCKS)
# -----------------------------
if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = ["AAPL", "TSLA", "NVDA"]

# -----------------------------
# STOCK LIST (SEARCHABLE)
# -----------------------------
all_stocks = [
    "AAPL","MSFT","TSLA","NVDA","AMZN","META","GOOGL","AMD","NFLX","INTC",
    "DIS","PYPL","SHOP","UBER","LYFT","BABA","ORCL","CRM","ADBE","QCOM"
]

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
page = st.sidebar.radio("📄 Pages", ["📊 Scanner", "🔍 Stock Selector"])

# -----------------------------
# FUNCTIONS
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

def get_company_name(ticker):
    try:
        return yf.Ticker(ticker).info.get("longName", ticker)
    except:
        return ticker

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

    df_clean['target_1d'] = (df_clean['Close'].shift(-1) > df_clean['Close']).astype(int)
    df_clean['target_3d'] = (df_clean['Close'].shift(-3) > df_clean['Close']).astype(int)
    df_clean['target_5d'] = (df_clean['Close'].shift(-5) > df_clean['Close']).astype(int)

    return df_clean.dropna()

def train_models(df):
    X = df[['Close', 'Volume', 'rsi', 'macd']]
    models = {}

    for label in ['target_1d', 'target_3d', 'target_5d']:
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, df[label])
        models[label] = model

    return models

def predict(models, df):
    latest = df[['Close', 'Volume', 'rsi', 'macd']].iloc[-1:]

    return {
        "1 Day": models['target_1d'].predict_proba(latest)[0][1],
        "3 Day": models['target_3d'].predict_proba(latest)[0][1],
        "5 Day": models['target_5d'].predict_proba(latest)[0][1]
    }

# -----------------------------
# PAGE 1: SCANNER
# -----------------------------
if page == "📊 Scanner":

    st.title("🚀 AI Stock Scanner")
    st.subheader("Your Selected Stocks")

    if st.button("🔍 Scan Selected Stocks"):

        cols = st.columns(3)

        for i, ticker in enumerate(st.session_state.selected_stocks):
            try:
                df = get_stock_data(ticker)

                if df is None:
                    continue

                models = train_models(df)
                probs = predict(models, df)

                price = df['Close'].iloc[-1]
                company = get_company_name(ticker)

                with cols[i % 3]:
                    st.markdown(f"### 📊 {company}")
                    st.caption(ticker)

                    st.metric("💰 Price", f"${price:.2f}")

                    for label, prob in probs.items():
                        color = get_color(prob)
                        st.markdown(
                            f"<span style='color:{color}; font-size:18px;'>📅 {label}: {prob:.2%}</span>",
                            unsafe_allow_html=True
                        )

                    avg_prob = sum(probs.values()) / 3
                    if avg_prob > 0.65:
                        signal = "🟢 Good Chance"
                    elif avg_prob > 0.5:
                        signal = "🟡 Maybe"
                    else:
                        signal = "🔴 Low Chance"

                    st.write(f"🧠 {signal}")

                    st.line_chart(df["Close"])

                    with st.expander("ℹ️ Simple Explanation"):
                        rsi = df['rsi'].iloc[-1]
                        macd = df['macd'].iloc[-1]

                        if rsi > 70:
                            st.write("📊 Stock went up a lot → might slow down")
                        elif rsi < 30:
                            st.write("📊 Stock dropped a lot → might go up")
                        else:
                            st.write("📊 Normal movement")

                        if macd > 0:
                            st.write("📈 Trend is going UP")
                        else:
                            st.write("📉 Trend is going DOWN")

                    st.divider()

            except:
                st.write(f"{ticker} error")

# -----------------------------
# PAGE 2: STOCK SELECTOR
# -----------------------------
elif page == "🔍 Stock Selector":

    st.title("🔍 Pick Your Stocks")

    search = st.text_input("Search stock ticker")

    filtered = [s for s in all_stocks if search.upper() in s]

    selected = []

    for stock in filtered:
        if st.checkbox(stock, value=(stock in st.session_state.selected_stocks)):
            selected.append(stock)

    # SAVE BUTTON
    if st.button("💾 Save Selection"):
        st.session_state.selected_stocks = selected
        st.success("Saved! Go back to Scanner 🚀")
