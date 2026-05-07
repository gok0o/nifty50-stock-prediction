import joblib
import numpy as np
import pandas as pd



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


prev_return = float(input(
    "Previous Return: "
))

prev_rsi = float(input(
    "Previous RSI: "
))

prev_target = int(input(
    "Previous Target (0/1): "
))

gap = float(input(
    "Gap: "
))

prev_index_return = float(input(
    "Previous NIFTY Return: "
))


# -----------------------------------
# STOCK ENCODING
# -----------------------------------

stock_id = encoder.transform(

    [stock_name]

)[0]


# -----------------------------------
# CREATE INPUT
# -----------------------------------


import pandas as pd


# -----------------------------------
# CREATE INPUT
# -----------------------------------

technical_features = pd.DataFrame([{

    "Prev_Return": prev_return,
    "Prev_RSI": prev_rsi,
    "Prev_Target": prev_target,
    "Gap": gap,
    "Prev_Index_Return": prev_index_return

}])


# Scale
scaled_array = scaler.transform(
    technical_features
)


# Convert back to DataFrame
scaled_df = pd.DataFrame(

    scaled_array,

    columns=[
        "Prev_Return",
        "Prev_RSI",
        "Prev_Target",
        "Gap",
        "Prev_Index_Return"
    ]

)


# Add stock id
scaled_df["Stock_ID"] = stock_id


# Final input
final_input = scaled_df


# -----------------------------------
# PREDICT
# -----------------------------------

prediction = model.predict(
    final_input
)[0]


probabilities = model.predict_proba(
    final_input
)[0]


# -----------------------------------
# RESULT
# -----------------------------------

print("\n" + "=" * 50)

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