import pandas as pd
import numpy as np
import yfinance as yf


def apply_indicators(group, index_df, vix_df):

    # -----------------------------------
    # SORT
    # -----------------------------------

    group = group.sort_values("Date")


    # -----------------------------------
    # BASE INDICATORS
    # -----------------------------------

    group["Return"] = group["Close"].pct_change()

    group["SMA_10"] = (
        group["Close"]
        .rolling(window=10)
        .mean()
    )


    # RSI

    delta = group["Close"].diff()

    gain = (
        delta.where(delta > 0, 0)
        .rolling(window=14)
        .mean()
    )

    loss = (
        -delta.where(delta < 0, 0)
        .rolling(window=14)
        .mean()
    )

    group["RSI"] = (
        100 - (100 / (1 + (gain / (loss + 1e-9))))
    )


    # Relative Volume

    group["Rel_Volume"] = (
        group["Volume"]
        / group["Volume"]
        .rolling(window=5)
        .mean()
    )


    # ATR

    prev_close = group["Close"].shift(1)

    tr1 = (
        group["High"]
        - group["Low"]
    )

    tr2 = abs(
        group["High"]
        - prev_close
    )

    tr3 = abs(
        group["Low"]
        - prev_close
    )

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    group["ATR"] = (
        true_range
        .rolling(window=14)
        .mean()
    )


    # Range

    group["Range"] = (
        (group["High"] - group["Low"])
        / group["Open"]
    )

    group["Prev_Range"] = (
        group["Range"].shift(1)
    )


    # Body Strength

    group["Body_Strength"] = (
        (group["Close"] - group["Open"])
        / group["Open"]
    )

    group["Prev_Body_Strength"] = (
        group["Body_Strength"].shift(1)
    )


    # Distance from SMA

    group["Dist_SMA"] = (
        (group["Close"] - group["SMA_10"])
        / group["SMA_10"]
    )

    group["Prev_Dist_SMA"] = (
        group["Dist_SMA"].shift(1)
    )


    # -----------------------------------
    # TARGET
    # -----------------------------------

    group["Target"] = (
        group["Close"] > group["Open"]
    ).astype(int)


    # -----------------------------------
    # LAG FEATURES
    # -----------------------------------

    group["Prev_Return"] = (
        group["Return"].shift(1)
    )

    group["Prev_RSI"] = (
        group["RSI"].shift(1)
    )

    group["Prev_Target"] = (
        group["Target"].shift(1)
    )

    group["Gap"] = (
        (group["Open"] - group["Close"].shift(1))
        / group["Close"].shift(1)
    )

    group["Prev_Rel_Volume"] = (
        group["Rel_Volume"].shift(1)
    )

    group["Prev_ATR"] = (
        group["ATR"].shift(1)
    )


    # -----------------------------------
    # MERGE NIFTY
    # -----------------------------------

    group = group.merge(
        index_df,
        on="Date",
        how="left"
    )

    group["Prev_Index_Return"] = (
        group["Return_Index"].shift(1)
    )


    # -----------------------------------
    # MERGE VIX
    # -----------------------------------

    group = group.merge(
        vix_df,
        on="Date",
        how="left"
    )


    # -----------------------------------
    # CLEAN
    # -----------------------------------

    group = group.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return group.dropna()



def process_data():

    # -----------------------------------
    # LOAD STOCK DATA
    # -----------------------------------

    df = pd.read_csv(
        "nifty_data.csv",
        parse_dates=["Date"]
    )


    # -----------------------------------
    # LOAD INDEX DATA
    # -----------------------------------

    idx_df = pd.read_csv(
        "nifty_index.csv",
        parse_dates=["Date"]
    )

    idx_df = idx_df.sort_values("Date")

    idx_df["Return_Index"] = (
        idx_df["Close"].pct_change()
    )

    idx_df = idx_df[
        ["Date", "Return_Index"]
    ]


    # -----------------------------------
    # LOAD VIX
    # -----------------------------------

    vix_df = yf.download(
        "^INDIAVIX",
        period="10y",
        progress=False
    )
    # Fix yfinance MultiIndex columns
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = (
            vix_df.columns.get_level_values(0)
        )

    vix_df = vix_df.reset_index()

    vix_df["VIX_Return"] = (
        vix_df["Close"].pct_change()
    )

    vix_df["Prev_VIX"] = (
        vix_df["Close"].shift(1)
    )

    vix_df = vix_df[
        ["Date", "VIX_Return", "Prev_VIX"]
    ]


    print(
        "Processing 50 stocks..."
    )


    processed_list = []

    for stock_name, group in df.groupby("Stock"):

        processed_group = apply_indicators(
            group,
            idx_df,
            vix_df
        )

        processed_group["Stock"] = stock_name

        processed_list.append(
            processed_group
        )


    processed_df = pd.concat(
        processed_list,
        ignore_index=True
    )


    processed_df.to_csv(
        "nifty_processed.csv",
        index=False
    )

    print(
        "Done! nifty_processed.csv created."
    )


if __name__ == "__main__":
    process_data()