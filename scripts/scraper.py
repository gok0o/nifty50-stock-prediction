import yfinance as yf
import pandas as pd


# -----------------------------------
# CURRENT NIFTY 50 CONSTITUENTS
# -----------------------------------

stocks = [

    "KOTAKBANK.NS",
    "ICICIBANK.NS",
    "ADANIENT.NS",
    "HDFCBANK.NS",
    "AXISBANK.NS",
    "BHARTIARTL.NS",
    "RELIANCE.NS",
    "MARUTI.NS",
    "ADANIPORTS.NS",
    "SBIN.NS",
    "BAJAJ-AUTO.NS",
    "ETERNAL.NS",
    "LT.NS",
    "BAJFINANCE.NS",
    "M&M.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "INFY.NS",
    "TCS.NS",
    "SUNPHARMA.NS",
    "BEL.NS",
    "COALINDIA.NS",
    "EICHERMOT.NS",
    "SHRIRAMFIN.NS",

    # Recent additions
    "INDIGO.NS",
    "MAXHEALTH.NS",

    "HINDALCO.NS",
    "TATASTEEL.NS",
    "ULTRACEMCO.NS",
    "DRREDDY.NS",
    "JIOFIN.NS",
    "TITAN.NS",
    "ONGC.NS",
    "HCLTECH.NS",
    "BAJAJFINSV.NS",
    "APOLLOHOSP.NS",
    "SBILIFE.NS",
    "POWERGRID.NS",
    "NESTLEIND.NS",
    "JSWSTEEL.NS",
    "ASIANPAINT.NS",
    "NTPC.NS",

    # Recent addition
    "TRENT.NS",

    "WIPRO.NS",
    "TECHM.NS",
    "GRASIM.NS",
    "HDFCLIFE.NS",
    "TATACONSUM.NS",
    "CIPLA.NS",

    # Sometimes fails in Yahoo
    "TATAMOTORS.NS"
]


# -----------------------------------
# SCRAPE STOCK DATA
# -----------------------------------

all_data = []
failed_stocks = []

for stock in stocks:

    print(f"Scraping {stock}...")

    try:

        df = yf.download(
            stock,
            start="2014-01-01",
            end="2025-12-31",
            interval="1d",
            auto_adjust=False
        )

        if df.empty:

            print(f"Skipping {stock}")
            failed_stocks.append(stock)
            continue

        # Flatten multi-index columns
        df.columns = df.columns.get_level_values(0)

        # Reset date index
        df = df.reset_index()

        # Add stock name
        df["Stock"] = stock

        # Store
        all_data.append(df)

    except Exception as e:

        print(f"Failed: {stock} -> {e}")

        failed_stocks.append(stock)

        continue


# -----------------------------------
# MERGE + SAVE STOCK DATA
# -----------------------------------

final_df = pd.concat(
    all_data,
    ignore_index=True
)

final_df.to_csv(
    "nifty_data.csv",
    index=False
)

print("\nStock data saved.")


# -----------------------------------
# DOWNLOAD NIFTY INDEX
# -----------------------------------

print("\nDownloading NIFTY index...")

try:

    nifty = yf.download(
        "^NSEI",
        start="2014-01-01",
        end="2025-12-31",
        interval="1d",
        auto_adjust=False
    )

    nifty.columns = nifty.columns.get_level_values(0)

    nifty = nifty.reset_index()

    nifty.to_csv(
        "nifty_index.csv",
        index=False
    )

    print("NIFTY index saved.")

except Exception as e:

    print(
        f"NIFTY download failed: {e}"
    )


# -----------------------------------
# FINAL SUMMARY
# -----------------------------------

print("\n" + "=" * 60)
print("SCRAPING COMPLETE")
print("=" * 60)

print(
    f"Final Shape: {final_df.shape}"
)

print(
    f"\nSuccessfully scraped: {len(all_data)}"
)

print(
    f"Failed: {failed_stocks}"
)

print(
    "\nSample Data:"
)

print(
    final_df.head()
)