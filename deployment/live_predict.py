import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import ta


# -----------------------------------
# LOAD SAVED FILES
# -----------------------------------

model = joblib.load(
    "lightgbm_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

encoder = joblib.load(
    "stock_encoder.pkl"
)


# -----------------------------------
# USER INPUT
# -----------------------------------

stock_name = input(
    "Enter stock (example RELIANCE.NS): "
).upper()


# -----------------------------------
# DOWNLOAD STOCK DATA
# -----------------------------------

stock_df = yf.download(

    stock_name,

    period="3mo",

    interval="1d",

    auto_adjust=False

)

stock_df.columns = stock_df.columns.get_level_values(0)

stock_df = stock_df.reset_index()


# -----------------------------------
# DOWNLOAD NIFTY DATA
# -----------------------------------

nifty_df = yf.download(

    "^NSEI",

    period="3mo",

    interval="1d"

)

nifty_df.columns = nifty_df.columns.get_level_values(0)

nifty_df = nifty_df.reset_index()


# -----------------------------------
# STOCK FEATURES
# -----------------------------------

stock_df["Prev_Return"] = (

    stock_df["Close"]
    .pct_change()

)

stock_df["Prev_RSI"] = ta.momentum.RSIIndicator(

    stock_df["Close"]

).rsi()


stock_df["Prev_Target"] = (

    stock_df["Close"]
    >
    stock_df["Open"]

).astype(int)


stock_df["Gap"] = (

    (
        stock_df["Open"]
        -
        stock_df["Close"].shift(1)
    )

    /

    stock_df["Close"].shift(1)

)


# -----------------------------------
# NIFTY FEATURE
# -----------------------------------

nifty_df["Prev_Index_Return"] = (

    nifty_df["Close"]
    .pct_change()

)


# Latest market return
latest_nifty_return = nifty_df.iloc[-1][

    "Prev_Index_Return"

]



# -----------------------------------
# GET LATEST ROW
# -----------------------------------

latest = stock_df.iloc[-1]


prev_return = latest[
    "Prev_Return"
]

prev_rsi = latest[
    "Prev_RSI"
]

prev_target = latest[
    "Prev_Target"
]

gap = latest[
    "Gap"
]


# -----------------------------------
# ENCODE STOCK
# -----------------------------------

stock_id = encoder.transform(

    [stock_name]

)[0]


# -----------------------------------
# CREATE INPUT
# -----------------------------------

features = pd.DataFrame([{

    "Prev_Return": prev_return,
    "Prev_RSI": prev_rsi,
    "Prev_Target": prev_target,
    "Gap": gap,
    "Prev_Index_Return": latest_nifty_return

}])


# Scale
scaled = scaler.transform(
    features
)


scaled_df = pd.DataFrame(

    scaled,

    columns=features.columns

)


scaled_df["Stock_ID"] = stock_id


# -----------------------------------
# PREDICT
# -----------------------------------

prediction = model.predict(
    scaled_df
)[0]


probabilities = model.predict_proba(
    scaled_df
)[0]


# -----------------------------------
# RESULT
# -----------------------------------

print("\n" + "=" * 50)

print(
    f"Stock: {stock_name}"
)

if prediction == 1:

    print(
        "Prediction: UP"
    )

else:

    print(
        "Prediction: DOWN"
    )


print(
    f"Confidence: {max(probabilities)*100:.2f}%"
)

print("=" * 50)