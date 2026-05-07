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
    
    # --- 2. TARGET (Intraday) ---
    group['Target'] = (group['Close'] > group['Open']).astype(int)
    
    # --- 3. LAGGED FEATURES ---
    group['Prev_Return'] = group['Return'].shift(1)
    group['Prev_RSI'] = group['RSI'].shift(1)
    group['Prev_Target'] = group['Target'].shift(1)
    group['Gap'] = (group['Open'] - group['Close'].shift(1)) / group['Close'].shift(1)

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