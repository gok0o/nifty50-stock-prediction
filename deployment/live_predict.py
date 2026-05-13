import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import ta


# =====================================
# LOAD MODEL FILES
# =====================================

model = joblib.load(
    "lightgbm_model.pkl"
)

encoder = joblib.load(
    "stock_encoder.pkl"
)


# =====================================
# MAPS
# =====================================

sector_map = {

    "HDFCBANK.NS": "BANK",
    "ICICIBANK.NS": "BANK",

    "TCS.NS": "IT",
    "INFY.NS": "IT",

    "MARUTI.NS": "AUTO",
    "M&M.NS": "AUTO"
}


peer_map = {

    "TCS.NS": "INFY.NS",
    "INFY.NS": "TCS.NS",

    "HDFCBANK.NS": "ICICIBANK.NS",
    "ICICIBANK.NS": "HDFCBANK.NS",

    "MARUTI.NS": "M&M.NS",
    "M&M.NS": "MARUTI.NS"
}


sector_tickers = {

    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "AUTO": "^CNXAUTO"
}


# =====================================
# USER INPUT
# =====================================

stock_name = input(
    "Enter stock (example RELIANCE.NS): "
).upper()


# =====================================
# DOWNLOAD STOCK
# =====================================

stock_df = yf.download(

    stock_name,

    period="3mo",

    interval="1d",

    auto_adjust=False

)

if isinstance(
    stock_df.columns,
    pd.MultiIndex
):
    stock_df.columns = (
        stock_df.columns
        .get_level_values(0)
    )

stock_df = stock_df.reset_index()


# =====================================
# DOWNLOAD NIFTY
# =====================================

nifty_df = yf.download(

    "^NSEI",

    period="3mo"

)

if isinstance(
    nifty_df.columns,
    pd.MultiIndex
):
    nifty_df.columns = (
        nifty_df.columns
        .get_level_values(0)
    )

nifty_df = nifty_df.reset_index()


# =====================================
# DOWNLOAD VIX
# =====================================

vix_df = yf.download(

    "^INDIAVIX",

    period="3mo"

)

if isinstance(
    vix_df.columns,
    pd.MultiIndex
):
    vix_df.columns = (
        vix_df.columns
        .get_level_values(0)
    )

vix_df = vix_df.reset_index()


# =====================================
# SECTOR DATA
# =====================================

sector_name = sector_map.get(
    stock_name
)

sector_return = 0
prev_sector = 0

if sector_name:

    sector_df = yf.download(

        sector_tickers[
            sector_name
        ],

        period="3mo"

    )

    if isinstance(
        sector_df.columns,
        pd.MultiIndex
    ):
        sector_df.columns = (
            sector_df.columns
            .get_level_values(0)
        )

    sector_df = sector_df.reset_index()

    sector_df[
        "Return"
    ] = (
        sector_df["Close"]
        .pct_change()
    )

    sector_return = (
        sector_df.iloc[-1]["Return"]
    )

    prev_sector = (
        sector_df.iloc[-2]["Close"]
    )


# =====================================
# PEER DATA
# =====================================

peer_return = 0

peer_stock = peer_map.get(
    stock_name
)

if peer_stock:

    peer_df = yf.download(

        peer_stock,

        period="3mo"

    )

    if isinstance(
        peer_df.columns,
        pd.MultiIndex
    ):
        peer_df.columns = (
            peer_df.columns
            .get_level_values(0)
        )

    peer_df = peer_df.reset_index()

    peer_df[
        "Return"
    ] = (
        peer_df["Close"]
        .pct_change()
    )

    peer_return = (
        peer_df.iloc[-1]["Return"]
    )


# =====================================
# CREATE FEATURES
# =====================================

stock_df["Return"] = (
    stock_df["Close"]
    .pct_change()
)

stock_df["SMA_10"] = (
    stock_df["Close"]
    .rolling(10)
    .mean()
)

stock_df["RSI"] = ta.momentum.RSIIndicator(
    stock_df["Close"]
).rsi()


stock_df["Rel_Volume"] = (

    stock_df["Volume"]

    /

    stock_df["Volume"]
    .rolling(5)
    .mean()

)


stock_df["ATR"] = ta.volatility.AverageTrueRange(

    stock_df["High"],
    stock_df["Low"],
    stock_df["Close"]

).average_true_range()


stock_df["Range"] = (

    (
        stock_df["High"]
        -
        stock_df["Low"]
    )

    /

    stock_df["Open"]

)


stock_df["Body_Strength"] = (

    (
        stock_df["Close"]
        -
        stock_df["Open"]
    )

    /

    stock_df["Open"]

)


stock_df["Dist_SMA"] = (

    (
        stock_df["Close"]
        -
        stock_df["SMA_10"]
    )

    /

    stock_df["SMA_10"]

)


latest = stock_df.iloc[-1]


# =====================================
# MARKET FEATURES
# =====================================

nifty_df[
    "Return"
] = (
    nifty_df["Close"]
    .pct_change()
)

vix_df[
    "Return"
] = (
    vix_df["Close"]
    .pct_change()
)


# =====================================
# ENCODE STOCK
# =====================================

stock_id = encoder.transform(

    [stock_name]

)[0]


# =====================================
# MODEL INPUT
# =====================================

X = pd.DataFrame([{

    "Prev_Return": latest["Return"],
    "Prev_RSI": latest["RSI"],
    "Gap": (
        (
            latest["Open"]
            -
            stock_df.iloc[-2]["Close"]
        )
        /
        stock_df.iloc[-2]["Close"]
    ),

    "Prev_Index_Return":
        nifty_df.iloc[-1]["Return"],

    "Prev_Rel_Volume":
        latest["Rel_Volume"],

    "Prev_ATR":
        latest["ATR"],

    "Prev_Range":
        latest["Range"],

    "Prev_Body_Strength":
        latest["Body_Strength"],

    "Prev_Dist_SMA":
        latest["Dist_SMA"],

    "VIX_Return":
        vix_df.iloc[-1]["Return"],

    "Prev_VIX":
        vix_df.iloc[-2]["Close"],

    "Sector_Return":
        sector_return,

    "Prev_Sector":
        prev_sector,

    "Peer_Return":
        peer_return,

    "Stock_ID":
        stock_id

}])


# =====================================
# PREDICT
# =====================================

prediction = model.predict(
    X
)[0]

prob = model.predict_proba(
    X
)[0]


# =====================================
# OUTPUT
# =====================================

print("\n" + "="*50)

print(
    f"Stock: {stock_name}"
)

print(
    f"Prediction: {'UP' if prediction == 1 else 'DOWN'}"
)

print(
    f"Confidence: {max(prob)*100:.2f}%"
)

print("="*50)