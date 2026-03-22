import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

st.title("📈 AI Stock Predictor (1 Week Max Hold)")

def get_stock_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")

    df['rsi'] = RSIIndicator(df['Close']).rsi()
    macd = MACD(df['Close'])
    df['macd'] = macd.macd()

    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna()

def train_model(df):
    X = df[['Close', 'Volume', 'rsi', 'macd']]
    y = df['target']

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

def predict(model, df):
    latest = df[['Close', 'Volume', 'rsi', 'macd']].iloc[-1:]
    prob = model.predict_proba(latest)[0][1]
    return prob

def strategy(price):
    return {
        "Sell (Take Profit)": round(price * 1.05, 2),
        "Stop Loss": round(price * 0.98, 2),
        "Max Hold Days": 5
    }

ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Analyze"):
    df = get_stock_data(ticker)
    model = train_model(df)

    prob = predict(model, df)
    price = df['Close'].iloc[-1]

    st.subheader(f"{ticker} Results")
    st.write(f"💰 Price: ${price:.2f}")
    st.write(f"📊 Chance of Going Up: {prob:.2%}")
    st.write(strategy(price))
