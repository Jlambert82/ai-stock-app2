import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="AI Stock Scanner", layout="wide")

# -----------------------------
# SESSION STATE
# -----------------------------
if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = ["AAPL", "TSLA", "NVDA"]

# -----------------------------
# STOCK LIST
# -----------------------------
all_stocks = [
    "AAPL","MSFT","TSLA","NVDA","AMZN","META","GOOGL","AMD","NFLX","INTC",
    "DIS","PYPL","SHOP","UBER","LYFT","BABA","ORCL","CRM","ADBE","QCOM"
]

# -----------------------------
# SIDEBAR
# -----------------------------
page = st.sidebar.radio("📄 Pages", ["📊 Scanner", "🔍 Stock Selector"])

# -----------------------------
# HELPERS
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

# -----------------------------
# DATA
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

    df_clean['target_1d'] = (df_clean['Close'].shift(-1) > df_clean['Close']).astype(int)
    df_clean['target_3d'] = (df_clean['Close'].shift(-3) > df_clean['Close']).astype(int)
    df_clean['target_5d'] = (df_clean['Close'].shift(-5) > df_clean['Close']).astype(int)

    return df_clean.dropna()

# -----------------------------
# MODELS
# -----------------------------
def train_models(df):
    X = df[['Close', 'Volume', 'rsi', 'macd']]

    models = {}
    regressors = {}

    for days, label in zip([1, 3, 5], ['target_1d', 'target_3d', 'target_5d']):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, df[label])
        models[label] = clf

        future_price = df['Close'].shift(-days)
        pct_change = (future_price - df['Close']) / df['Close']

        reg = RandomForestRegressor(n_estimators=100)
        reg.fit(X, pct_change.fillna(0))
        regressors[label] = reg

    return models, regressors

# -----------------------------
# PREDICT
# -----------------------------
def predict(models, regressors, df):
    latest = df[['Close', 'Volume', 'rsi', 'macd']].iloc[-1:]
    current_price = df['Close'].iloc[-1]

    results = {}

    for label, days in zip(['target_1d','target_3d','target_5d'], [1,3,5]):
        prob = models[label].predict_proba(latest)[0][1]
        pct = regressors[label].predict(latest)[0]
        expected_price = current_price * (1 + pct)

        results[days] = {
            "prob": prob,
            "pct": pct,
            "price": expected_price
        }

    return results

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

                models, regressors = train_models(df)
                preds = predict(models, regressors, df)

                price = df['Close'].iloc[-1]
                company = get_company_name(ticker)

                with cols[i % 3]:
                    st.markdown(f"### 📊 {company}")
                    st.caption(ticker)

                    st.metric("💰 Price", f"${price:.2f}")

                    # Predictions
                    for days, data in preds.items():
                        prob = data["prob"]
                        pct = data["pct"]
                        target = data["price"]

                        color = get_color(prob)

                        st.markdown(
                            f"<span style='color:{color}; font-size:18px;'>"
                            f"📅 {days} Day: {prob:.2%} | {pct:+.2%} → ${target:.2f}"
                            f"</span>",
                            unsafe_allow_html=True
                        )

                    avg_prob = sum([d["prob"] for d in preds.values()]) / 3

                    if avg_prob > 0.65:
                        signal = "🟢 Strong Opportunity"
                    elif avg_prob > 0.5:
                        signal = "🟡 Moderate Opportunity"
                    else:
                        signal = "🔴 Weak Setup"

                    st.write(f"🧠 {signal}")

                    st.line_chart(df["Close"])

                    # -----------------------------
                    # DETAILED EXPLANATION
                    # -----------------------------
                    with st.expander("ℹ️ Detailed Analysis"):
                        rsi = df['rsi'].iloc[-1]
                        macd = df['macd'].iloc[-1]

                        st.write("🧠 **AI Breakdown:**")
                        st.write("This model uses price trends, momentum, and trading activity to predict short-term movement.")

                        st.write("---")

                        st.write(f"📊 **RSI (Momentum): {rsi:.2f}**")
                        if rsi > 70:
                            st.write("• Overbought → price ran up fast")
                            st.write("• Higher risk of pullback")
                        elif rsi < 30:
                            st.write("• Oversold → price dropped a lot")
                            st.write("• Possible bounce opportunity")
                        else:
                            st.write("• Neutral momentum")
                            st.write("• No extreme pressure")

                        st.write("---")

                        st.write(f"📈 **MACD (Trend): {macd:.2f}**")
                        if macd > 0:
                            st.write("• Bullish trend")
                            st.write("• Buyers currently in control")
                        else:
                            st.write("• Bearish trend")
                            st.write("• Sellers currently in control")

                        st.write("---")

                        st.write("📅 **Prediction Windows:**")
                        st.write("• 1 Day → Immediate move")
                        st.write("• 3 Day → Short trend")
                        st.write("• 5 Day → Weekly direction")

                        st.write("---")

                        st.write("🎯 **How to Read This:**")
                        st.write("• High % + positive price = strong setup")
                        st.write("• Low % but big upside = risky play")
                        st.write("• Red + negative = avoid")

                    st.divider()

            except Exception:
                st.write(f"{ticker} error")

# -----------------------------
# PAGE 2: SELECTOR
# -----------------------------
elif page == "🔍 Stock Selector":

    st.title("🔍 Pick Your Stocks")

    search = st.text_input("Search ticker")

    filtered = [s for s in all_stocks if search.upper() in s]

    selected = []

    for stock in filtered:
        if st.checkbox(stock, value=(stock in st.session_state.selected_stocks)):
            selected.append(stock)

    if st.button("💾 Save Selection"):
        st.session_state.selected_stocks = selected
        st.success("Saved! Go back to Scanner 🚀")
