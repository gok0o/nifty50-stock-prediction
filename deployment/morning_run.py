import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import ta
import os
from datetime import date


# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("stock_encoder.pkl")


# -----------------------------------
# STOCK LIST
# -----------------------------------

stocks = pd.read_csv("nifty_data.csv")["Stock"].unique().tolist()
print("Total stocks:", len(stocks))


# -----------------------------------
# DATE
# -----------------------------------

today = str(date.today())


# -----------------------------------
# NIFTY DATA
# -----------------------------------

nifty = yf.download("^NSEI", period="3mo", progress=False)

if nifty is None or nifty.empty:
    print("Failed to fetch NIFTY data")
    exit()

if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)

nifty["Prev_Index_Return"] = nifty["Close"].pct_change()
nifty = nifty.dropna()

nifty_return = nifty.iloc[-1]["Prev_Index_Return"]

print("NIFTY return:", nifty_return)


# -----------------------------------
# LOOP
# -----------------------------------

rows = []

for stock in stocks:

    print(f"Processing: {stock}")

    try:
        stock_df = yf.download(
            stock,
            period="3mo",
            interval="1d",
            progress=False
        )

        if stock_df is None or stock_df.empty:
            print(f"Skipping {stock}")
            continue

        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)

        stock_df = stock_df.reset_index()

        # FEATURES
        stock_df["Prev_Return"] = stock_df["Close"].pct_change()
        stock_df["Prev_RSI"] = ta.momentum.RSIIndicator(
            stock_df["Close"]
        ).rsi()

        stock_df["Prev_Target"] = (
            stock_df["Close"] > stock_df["Open"]
        ).astype(int)

        stock_df["Gap"] = (
            (stock_df["Open"] - stock_df["Close"].shift(1))
            / stock_df["Close"].shift(1)
        )

        stock_df = stock_df.replace([np.inf, -np.inf], np.nan)
        stock_df = stock_df.dropna()

        if len(stock_df) == 0:
            print(f"No valid data for {stock}")
            continue

        latest = stock_df.iloc[-1]

        # INPUT
        features = pd.DataFrame([{
            "Prev_Return": latest["Prev_Return"],
            "Prev_RSI": latest["Prev_RSI"],
            "Prev_Target": latest["Prev_Target"],
            "Gap": latest["Gap"],
            "Prev_Index_Return": nifty_return
        }])

        scaled = scaler.transform(features)

        scaled_df = pd.DataFrame(
            scaled,
            columns=features.columns
        )

        stock_id = encoder.transform([stock])[0]
        scaled_df["Stock_ID"] = stock_id

        pred = model.predict(scaled_df)[0]
        prob = model.predict_proba(scaled_df)[0]

        rows.append({
            "Date": today,
            "Stock": stock,
            "Open": latest["Open"],
            "Prediction": "UP" if pred == 1 else "DOWN",
            "Confidence": round(max(prob), 4),
            "Close": np.nan,
            "Actual": None
        })

        print(f"Added: {stock}")

    except Exception as e:
        print(f"Error: {stock} -> {e}")
        continue


file_path = "predictions_log.csv"

# Define correct column order
columns = [
    "Date",
    "Stock",
    "Open",
    "Prediction",
    "Confidence",
    "Close",
    "Actual"
]

df_out = pd.DataFrame(rows)[columns]

# -----------------------------------
# FIX HEADER ISSUE
# -----------------------------------

if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
    # File doesn't exist OR empty → write with header
    df_out.to_csv(file_path, index=False)

else:
    # File exists → check if header is missing
    existing_df = pd.read_csv(file_path, header=None)

    if existing_df.shape[1] != len(columns):
        print("Fixing corrupted CSV...")

        # Rewrite file with header
        df_out.to_csv(file_path, index=False)

    else:
        # Normal append
        df_out.to_csv(file_path, mode='a', header=False, index=False)