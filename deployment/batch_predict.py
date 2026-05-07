import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import ta


# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("stock_encoder.pkl")


# -----------------------------------
# STOCK LIST (you can edit this)
# -----------------------------------

stocks = [

    "RELIANCE.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "TATAMOTORS.NS",
    "BHARTIARTL.NS",
    "ICICIBANK.NS",
    "AXISBANK.NS",
    "ADANIENT.NS",
    "MARUTI.NS",
    "TCS.NS"

]


# -----------------------------------
# DOWNLOAD NIFTY DATA ONCE
# -----------------------------------

nifty_df = yf.download(
    "^NSEI",
    period="3mo",
    interval="1d"
)

nifty_df.columns = nifty_df.columns.get_level_values(0)
nifty_df = nifty_df.reset_index()

nifty_df["Prev_Index_Return"] = nifty_df["Close"].pct_change()

latest_nifty_return = nifty_df.iloc[-1]["Prev_Index_Return"]


# -----------------------------------
# RESULTS
# -----------------------------------

results = []


# -----------------------------------
# LOOP
# -----------------------------------

for stock in stocks:

    try:

        stock_df = yf.download(
            stock,
            period="3mo",
            interval="1d"
        )

        if stock_df.empty:
            continue

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

        latest = stock_df.iloc[-1]

        # EXTRACT
        prev_return = latest["Prev_Return"]
        prev_rsi = latest["Prev_RSI"]
        prev_target = latest["Prev_Target"]
        gap = latest["Gap"]

        # SKIP if NaN
        if np.isnan(prev_return) or np.isnan(prev_rsi) or np.isnan(gap):
            continue

        # ENCODE
        stock_id = encoder.transform([stock])[0]

        # CREATE INPUT
        features = pd.DataFrame([{

            "Prev_Return": prev_return,
            "Prev_RSI": prev_rsi,
            "Prev_Target": prev_target,
            "Gap": gap,
            "Prev_Index_Return": latest_nifty_return

        }])

        scaled = scaler.transform(features)

        scaled_df = pd.DataFrame(
            scaled,
            columns=features.columns
        )

        scaled_df["Stock_ID"] = stock_id

        # PREDICT
        pred = model.predict(scaled_df)[0]
        prob = model.predict_proba(scaled_df)[0]

        results.append({

            "Stock": stock,
            "Prediction": "UP" if pred == 1 else "DOWN",
            "Confidence": round(max(prob)*100, 2)

        })

    except Exception as e:
        print(f"Error with {stock}: {e}")
        continue


# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Confidence",
    ascending=False
)

print("\n" + "="*60)
print("BATCH PREDICTIONS")
print("="*60)

print(results_df)