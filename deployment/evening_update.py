import pandas as pd
import yfinance as yf


# -----------------------------------
# LOAD CSV
# -----------------------------------

file_path = "predictions_log.csv"

df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())


# -----------------------------------
# FIX DATA TYPES (IMPORTANT)
# -----------------------------------

# Ensure text columns are not treated as float
df["Actual"] = df["Actual"].astype("object")
df["Prediction"] = df["Prediction"].astype("object")

# Ensure numeric columns stay numeric
df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")


# -----------------------------------
# GET LATEST DATE
# -----------------------------------

if "Date" not in df.columns:
    raise Exception("Column 'Date' not found in CSV")

latest_date = df["Date"].max()

print("\nUpdating data for:", latest_date)


# -----------------------------------
# FILTER TODAY'S ROWS
# -----------------------------------

mask = df["Date"] == latest_date


# -----------------------------------
# LOOP THROUGH STOCKS
# -----------------------------------

for i in df[mask].index:

    stock = df.loc[i, "Stock"]

    print(f"Processing: {stock}")

    try:
        data = yf.download(
            stock,
            period="1d",
            interval="1d",
            progress=False
        )

        # Skip if no data
        if data is None or data.empty:
            print(f"Skipping {stock} (no data)")
            continue

        # Fix multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        # -----------------------------------
        # EXTRACT OPEN + CLOSE
        # -----------------------------------

        open_price = data.iloc[-1]["Open"]
        close_price = data.iloc[-1]["Close"]

        # -----------------------------------
        # UPDATE VALUES
        # -----------------------------------

        df.loc[i, "Open"] = open_price
        df.loc[i, "Close"] = close_price

        df.loc[i, "Actual"] = (
            "UP" if close_price > open_price else "DOWN"
        )

        print(f"Updated: {stock}")

    except Exception as e:
        print(f"Error: {stock} -> {e}")
        continue


# -----------------------------------
# SAVE UPDATED CSV
# -----------------------------------

df.to_csv(file_path, index=False)

print("\nEvening update completed successfully.")