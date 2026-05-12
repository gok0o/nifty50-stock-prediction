import pandas as pd
import numpy as np

def apply_indicators(group, index_df):
    # Sort by date
    group = group.sort_values('Date')
    
    # --- 1. BASE INDICATORS ---
    group['Return'] = group['Close'].pct_change()
    group['SMA_10'] = group['Close'].rolling(window=10).mean()
    
    delta = group['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    group['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    # Relative Volume
    group['Rel_Volume'] = (
        group['Volume']
        / group['Volume'].rolling(window=5).mean()
    )
    # ATR calculation
    prev_close = group['Close'].shift(1)

    tr1 = group['High'] - group['Low']
    tr2 = abs(group['High'] - prev_close)
    tr3 = abs(group['Low'] - prev_close)

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    group['ATR'] = true_range.rolling(window=14).mean()
    # Previous day range
    group["Range"] = (
        (group["High"] - group["Low"])
        / group["Open"]
    )

    group["Prev_Range"] = (
        group["Range"].shift(1)
    )
# Previous candle body strength
    group["Body_Strength"] = (
        (group["Close"] - group["Open"])
        / group["Open"]
    )

    group["Prev_Body_Strength"] = (
        group["Body_Strength"].shift(1)
    )
    # Distance from moving average
    group["Dist_SMA"] = (
        (group["Close"] - group["SMA_10"])
        / group["SMA_10"]
    )

    group["Prev_Dist_SMA"] = (
        group["Dist_SMA"].shift(1)
    )
    
    # --- 2. TARGET (Intraday) ---
    group['Target'] = (group['Close'] > group['Open']).astype(int)
    
    # --- 3. LAGGED FEATURES ---
    group['Prev_Return'] = group['Return'].shift(1)
    group['Prev_RSI'] = group['RSI'].shift(1)
    group['Prev_Target'] = group['Target'].shift(1)
    group['Gap'] = (group['Open'] - group['Close'].shift(1)) / group['Close'].shift(1)
    group['Prev_Rel_Volume'] = group['Rel_Volume'].shift(1)
    group['Prev_ATR'] = group['ATR'].shift(1)

    # --- 4. MERGE INDEX DATA ---
    group = group.merge(index_df, on='Date', how='left', suffixes=('', '_Index'))
    group['Prev_Index_Return'] = group['Return_Index'].shift(1)
    

    # We do NOT drop the 'Stock' column here; it stays with the group
    return group.dropna()

def process_data():
    try:
        df = pd.read_csv("nifty_data.csv", parse_dates=['Date'])
        idx_df = pd.read_csv("nifty_index.csv", parse_dates=['Date'])
    except FileNotFoundError:
        print("Error: Ensure nifty_data.csv and nifty_index.csv exist.")
        return
    
    # Prepare Index Data
    idx_df = idx_df.sort_values('Date')
    idx_df['Return_Index'] = idx_df['Close'].pct_change()
    idx_df = idx_df[['Date', 'Return_Index']]
    
    print("Processing 50 stocks with global index features...")
    
    # --- THE FIX ---
    # We use a list for the groupby key and include it in the 'apply'
    # This prevents 'Stock' from disappearing or becoming a hidden attribute
    processed_list = []
    for stock_name, group in df.groupby('Stock'):
        # Pass the group to our function
        processed_group = apply_indicators(group, idx_df)
        # Ensure the stock name is explicitly in a column
        processed_group['Stock'] = stock_name
        processed_list.append(processed_group)
    
    # Combine everything back together
    processed_df = pd.concat(processed_list, ignore_index=True)

    # Save to CSV
    processed_df.to_csv("nifty_processed.csv", index=False)
    print("Done! 'nifty_processed.csv' created with stock identifiers.")

if __name__ == "__main__":
    process_data()