import pandas as pd
import joblib


# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "nifty_processed.csv",
    parse_dates=["Date"]
)


# ==========================
# LOAD MODEL + ENCODER
# ==========================

model = joblib.load(
    "lightgbm_model.pkl"
)

encoder = joblib.load(
    "stock_encoder.pkl"
)


# Encode stocks
df["Stock_ID"] = encoder.transform(
    df["Stock"]
)


# ==========================
# FEATURES
# ==========================

feature_cols = [

    "Prev_Return",
    "Prev_RSI",
    "Gap",
    "Prev_Index_Return",

    "Prev_Rel_Volume",
    "Prev_ATR",

    "Prev_Range",
    "Prev_Body_Strength",
    "Prev_Dist_SMA",

    "VIX_Return",
    "Prev_VIX",

    "Sector_Return",
    "Prev_Sector",

    "Peer_Return",

    "Stock_ID"
]


# ==========================
# LAST 10 DAYS
# ==========================

last_10_dates = sorted(
    df["Date"].unique()
)[-10:]

test_df = df[
    df["Date"].isin(last_10_dates)
].copy()


# ==========================
# PREDICTIONS
# ==========================

X_test = test_df[
    feature_cols
]

probs = model.predict_proba(
    X_test
)[:, 1]

threshold = 0.44

test_df["Prediction"] = (
    probs > threshold
).astype(int)

# Convert prediction to label
test_df["Predicted_Status"] = (
    test_df["Prediction"]
    .map({
        1: "UP",
        0: "DOWN"
    })
)


# ==========================
# ACTUAL STATUS
# ==========================

test_df["Actual_Status"] = (
    (test_df["Close"] > test_df["Open"])
    .map({
        True: "UP",
        False: "DOWN"
    })
)


# ==========================
# CORRECT / WRONG
# ==========================

test_df["Correct"] = (
    test_df["Predicted_Status"]
    == test_df["Actual_Status"]
)


accuracy = (
    test_df["Correct"]
    .mean()
)


# ==========================
# SAVE CSV
# ==========================

output = test_df[

    [
        "Date",
        "Stock",

        "Open",
        "Close",

        "Predicted_Status",
        "Actual_Status",

        "Correct"
    ]

]

output.to_csv(
    "last_10day_predictions.csv",
    index=False
)


# ==========================
# RESULTS
# ==========================

print("\n========== LAST 10 DAYS TEST ==========")

print(
    f"Rows Tested: {len(output)}"
)

print(
    f"Accuracy: {accuracy:.2%}"
)

print(
    "\nSaved:"
)

print(
    "last_10day_predictions.csv"
)