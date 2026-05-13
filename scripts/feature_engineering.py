import pandas as pd
import numpy as np
import yfinance as yf


# ==========================================
# STOCK → SECTOR MAP
# ==========================================

sector_map = {

    # BANK
    "HDFCBANK.NS": "BANK",
    "ICICIBANK.NS": "BANK",
    "AXISBANK.NS": "BANK",
    "SBIN.NS": "BANK",
    "KOTAKBANK.NS": "BANK",

    # IT
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "WIPRO.NS": "IT",
    "TECHM.NS": "IT",
    "HCLTECH.NS": "IT",

    # AUTO
    "MARUTI.NS": "AUTO",
    "M&M.NS": "AUTO",
    "BAJAJ-AUTO.NS": "AUTO"
}


# ==========================================
# STOCK → PEER MAP
# ==========================================

peer_map = {

    # IT
    "TCS.NS": "INFY.NS",
    "INFY.NS": "TCS.NS",
    "WIPRO.NS": "TCS.NS",
    "TECHM.NS": "INFY.NS",

    # BANK
    "HDFCBANK.NS": "ICICIBANK.NS",
    "ICICIBANK.NS": "HDFCBANK.NS",
    "AXISBANK.NS": "ICICIBANK.NS",
    "SBIN.NS": "HDFCBANK.NS",

    # AUTO
    "MARUTI.NS": "M&M.NS",
    "M&M.NS": "MARUTI.NS"
}


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def apply_indicators(
    group,
    index_df,
    vix_df,
    sector_df,
    peer_df
):

    group = group.sort_values("Date")


    # --------------------------------
    # BASE FEATURES
    # --------------------------------

    group["Return"] = (
        group["Close"].pct_change()
    )

    group["SMA_10"] = (
        group["Close"]
        .rolling(10)
        .mean()
    )


    # RSI

    delta = group["Close"].diff()

    gain = (
        delta.where(delta > 0, 0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.where(delta < 0, 0)
        .rolling(14)
        .mean()
    )

    group["RSI"] = (
        100 - (
            100 / (
                1 + (
                    gain / (loss + 1e-9)
                )
            )
        )
    )


    # Relative Volume

    group["Rel_Volume"] = (
        group["Volume"]
        / group["Volume"]
        .rolling(5)
        .mean()
    )


    # ATR

    prev_close = (
        group["Close"].shift(1)
    )

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
        .rolling(14)
        .mean()
    )


    # Candle Features

    group["Range"] = (
        (group["High"] - group["Low"])
        / group["Open"]
    )

    group["Body_Strength"] = (
        (group["Close"] - group["Open"])
        / group["Open"]
    )

    group["Dist_SMA"] = (
        (group["Close"] - group["SMA_10"])
        / group["SMA_10"]
    )


    # --------------------------------
    # TARGET
    # --------------------------------

    group["Target"] = (

        (
            (group["Close"] - group["Open"])
            / group["Open"]
        ) > 0.003

    ).astype(int)


    # --------------------------------
    # LAG FEATURES
    # --------------------------------

    group["Prev_Return"] = (
        group["Return"].shift(1)
    )

    group["Prev_RSI"] = (
        group["RSI"].shift(1)
    )

    group["Gap"] = (
        (
            group["Open"]
            - group["Close"].shift(1)
        )
        / group["Close"].shift(1)
    )

    group["Prev_Rel_Volume"] = (
        group["Rel_Volume"].shift(1)
    )

    group["Prev_ATR"] = (
        group["ATR"].shift(1)
    )

    group["Prev_Range"] = (
        group["Range"].shift(1)
    )

    group["Prev_Body_Strength"] = (
        group["Body_Strength"].shift(1)
    )

    group["Prev_Dist_SMA"] = (
        group["Dist_SMA"].shift(1)
    )


    # --------------------------------
    # MARKET FEATURES
    # --------------------------------

    group = group.merge(
        index_df,
        on="Date",
        how="left"
    )

    group["Prev_Index_Return"] = (
        group["Return_Index"].shift(1)
    )


    # --------------------------------
    # VIX FEATURES
    # --------------------------------

    group = group.merge(
        vix_df,
        on="Date",
        how="left"
    )


    # --------------------------------
    # SECTOR FEATURES
    # --------------------------------

    group = group.merge(
        sector_df,
        on="Date",
        how="left"
    )


    # --------------------------------
    # PEER FEATURES
    # --------------------------------

    group = group.merge(
        peer_df,
        on="Date",
        how="left"
    )


    # --------------------------------
    # CLEAN
    # --------------------------------

    group = group.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return group.dropna()



# ==========================================
# MAIN
# ==========================================

def process_data():

    df = pd.read_csv(
        "nifty_data.csv",
        parse_dates=["Date"]
    )


    # --------------------------------
    # NIFTY INDEX
    # --------------------------------

    idx_df = pd.read_csv(
        "nifty_index.csv",
        parse_dates=["Date"]
    )

    idx_df["Return_Index"] = (
        idx_df["Close"].pct_change()
    )

    idx_df = idx_df[
        ["Date", "Return_Index"]
    ]


    # --------------------------------
    # INDIA VIX
    # --------------------------------

    vix_df = yf.download(
        "^INDIAVIX",
        period="10y",
        progress=False
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

    vix_df["VIX_Return"] = (
        vix_df["Close"].pct_change()
    )

    vix_df["Prev_VIX"] = (
        vix_df["Close"].shift(1)
    )

    vix_df = vix_df[
        ["Date", "VIX_Return", "Prev_VIX"]
    ]


    # --------------------------------
    # SECTOR DATA
    # --------------------------------

    sector_tickers = {
        "BANK": "^NSEBANK",
        "IT": "^CNXIT",
        "AUTO": "^CNXAUTO"
    }

    sector_data = {}

    for sector_name, ticker in sector_tickers.items():

        temp = yf.download(
            ticker,
            period="10y",
            progress=False
        )

        if isinstance(
            temp.columns,
            pd.MultiIndex
        ):
            temp.columns = (
                temp.columns
                .get_level_values(0)
            )

        temp = temp.reset_index()

        temp["Sector_Return"] = (
            temp["Close"].pct_change()
        )

        temp["Prev_Sector"] = (
            temp["Close"].shift(1)
        )

        temp = temp[
            [
                "Date",
                "Sector_Return",
                "Prev_Sector"
            ]
        ]

        sector_data[
            sector_name
        ] = temp


    # --------------------------------
    # PEER DATA
    # --------------------------------

    peer_data = {}

    for stock_name, group in df.groupby("Stock"):

        temp = group.copy()

        temp = temp.sort_values("Date")

        temp["Peer_Return"] = (
            temp["Close"].pct_change()
        )

        peer_data[
            stock_name
        ] = temp[
            ["Date", "Peer_Return"]
        ]


    print(
        "Processing 50 stocks..."
    )


    processed_list = []


    for stock_name, group in df.groupby("Stock"):

        # Sector
        sector_name = sector_map.get(
            stock_name
        )

        if sector_name:

            sector_df = sector_data.get(
                sector_name
            )

        else:

            sector_df = pd.DataFrame({
                "Date": group["Date"],
                "Sector_Return": 0,
                "Prev_Sector": 0
            })


        # Peer
        peer_stock = peer_map.get(
            stock_name
        )

        if peer_stock:

            peer_df = peer_data.get(
                peer_stock
            )

        else:

            peer_df = pd.DataFrame({
                "Date": group["Date"],
                "Peer_Return": 0
            })


        processed_group = apply_indicators(
            group,
            idx_df,
            vix_df,
            sector_df,
            peer_df
        )

        processed_group["Stock"] = (
            stock_name
        )

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
        "Done! Features added."
    )


if __name__ == "__main__":
    process_data()