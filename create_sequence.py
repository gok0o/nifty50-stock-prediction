import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


WINDOW = 10


# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("nifty_processed.csv")

df = df.sort_values(
    ["Stock", "Date"]
).reset_index(drop=True)


# -----------------------------------
# FEATURES
# -----------------------------------

feature_cols = [
    "Prev_Return",
    "Prev_RSI",
    "Gap",
    "Prev_Index_Return",
    "Prev_Rel_Volume",
    "Prev_ATR",
    "Prev_Range",
    "Prev_Body_Strength",
    "Prev_Dist_SMA"
]

target_col = "Target"


# -----------------------------------
# SCALE FEATURES
# -----------------------------------

scaler = StandardScaler()

df[feature_cols] = scaler.fit_transform(
    df[feature_cols]
)

joblib.dump(
    scaler,
    "lstm_scaler.pkl"
)


# -----------------------------------
# CREATE SEQUENCES
# -----------------------------------

X = []
y = []


for stock in df["Stock"].unique():

    stock_df = df[
        df["Stock"] == stock
    ].copy()

    stock_features = stock_df[
        feature_cols
    ].values

    stock_targets = stock_df[
        target_col
    ].values

    for i in range(
        WINDOW,
        len(stock_df)
    ):

        sequence = stock_features[
            i-WINDOW:i
        ]

        target = stock_targets[i]

        X.append(sequence)
        y.append(target)


X = np.array(X)
y = np.array(y)


print("X shape:", X.shape)
print("y shape:", y.shape)


# -----------------------------------
# SAVE
# -----------------------------------

np.save(
    "X_lstm.npy",
    X
)

np.save(
    "y_lstm.npy",
    y
)

print("Sequences created successfully.")