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
page = st.sidebar.radio(
    "📄 Pages",
    ["📊 Scanner", "🏆 Best Opportunities", "🔍 Stock Selector"]
)

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
# DATA FUNCTION
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
    df_clean["rsi"] = RSIIndicator(close=df_clean["Close"]).rsi()
    macd = MACD(close=df_clean["Close"])
    df_clean["macd"] = macd.macd()

    # Targets
    df_clean["target_1d"] = (df_clean["Close"].shift(-1) > df_clean["Close"]).astype(int)
    df_clean["target_3d"] = (df_clean["Close"].shift(-3) > df_clean["Close"]).astype(int)
    df_clean["target_5d"] = (df_clean["Close"].shift(-5) > df_clean["Close"]).astype(int)

    return df_clean.dropna()

# -----------------------------
# MODEL TRAINING
# -----------------------------
def train_models(df):
    X = df[["Close", "Volume", "rsi", "macd"]]

    models = {}
    regressors = {}

    for days, label in zip([1, 3, 5], ["target_1d", "target_3d", "target_5d"]):

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, df[label])
        models[label] = clf

        future_price = df["Close"].shift(-days)
        pct_change = (future_price - df["Close"]) / df["Close"]

        reg = RandomForestRegressor(n_estimators=100)
        reg.fit(X, pct_change.fillna(0))
        regressors[label] = reg

    return models, regressors

# -----------------------------
# PREDICTION
# -----------------------------
def predict(models, regressors, df):
    latest = df[["Close", "Volume", "rsi", "macd"]].iloc[-1:]
    current_price = df["Close"].iloc[-1]

    results = {}

    for label, days in zip(["target_1d", "target_3d", "target_5d"], [1, 3, 5]):
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
# SCORING
# -----------------------------
def calculate_score(preds):
    score = 0
    for d in preds.values():
        score += d["prob"] * d["pct"]
    return score

# -----------------------------
# PAGE 1: SCANNER
# -----------------------------
if page == "📊 Scanner":

    st.title("🚀 AI Stock Scanner")

    if st.button("🔍 Scan Selected Stocks"):

        cols = st.columns(3)

        for i, ticker in enumerate(st.session_state.selected_stocks):

            try:
                df = get_stock_data(ticker)
                if df is None:
                    continue

                models, regressors = train_models(df)
                preds = predict(models, regressors, df)

                price = df["Close"].iloc[-1]
                company = get_company_name(ticker)

                with cols[i % 3]:
                    st.markdown(f"### 📊 {company}")
                    st.caption(ticker)

                    st.metric("💰 Price", f"${price:.2f}")

                    for days, data in preds.items():
                        prob = data["prob"]
                        pct = data["pct"]
                        target = data["price"]

                        color = get_color(prob)

                        st.markdown(
                            f"<span style='color:{color}; font-size:18px;'>"
                            f"{days}D: {prob:.2%} | {pct:+.2%} → ${target:.2f}"
                            f"</span>",
                            unsafe_allow_html=True
                        )

                    # Alerts
                    avg_prob = sum([d["prob"] for d in preds.values()]) / 3
                    avg_pct = sum([d["pct"] for d in preds.values()]) / 3

                    if avg_prob > 0.7 and avg_pct > 0:
                        st.success("🚨 BUY ALERT")
                    elif avg_pct < -0.03:
                        st.error("⚠️ SELL ALERT (Possible Drop)")
                    else:
                        st.info("ℹ️ No strong signal")

                    st.line_chart(df["Close"])
                    st.divider()

            except Exception as e:
                st.error(f"{ticker} error: {e}")

# -----------------------------
# PAGE 2: BEST OPPORTUNITIES
# -----------------------------
elif page == "🏆 Best Opportunities":

    st.title("🏆 Best Stock Opportunities")

    results = []

    for ticker in st.session_state.selected_stocks:
        try:
            df = get_stock_data(ticker)
            if df is None:
                continue

            models, regressors = train_models(df)
            preds = predict(models, regressors, df)

            score = calculate_score(preds)
            price = df["Close"].iloc[-1]

            results.append((ticker, score, preds, price))

        except:
            continue

    results = sorted(results, key=lambda x: x[1], reverse=True)

    for ticker, score, preds, price in results:
        st.markdown(f"### {ticker} — Score: {score:.4f}")
        st.write(f"💰 Current Price: ${price:.2f}")

        for d, data in preds.items():
            st.write(
                f"{d}D → {data['pct']:+.2%} | {data['prob']:.2%} "
                f"(Target: ${data['price']:.2f})"
            )

        st.divider()

# -----------------------------
# PAGE 3: STOCK SELECTOR
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
        st.success("Saved!")
