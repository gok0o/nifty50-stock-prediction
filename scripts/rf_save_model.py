import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier


# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv(
    "nifty_processed.csv"
)


# -----------------------------------
# STOCK ENCODER
# -----------------------------------

encoder = LabelEncoder()

df["Stock_ID"] = encoder.fit_transform(
    df["Stock"]
)


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
        "Prev_Dist_SMA",
        "VIX_Return",
        "Prev_VIX",
        "Sector_Return",
        "Prev_Sector",
        "Peer_Return"

    ]


target_col = "Target"


X = df[
    feature_cols
]

y = df[
    target_col
]


# -----------------------------------
# SCALE
# -----------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# Add stock id
X_final = np.column_stack((

    X_scaled,
    df["Stock_ID"].values

))


# -----------------------------------
# SMOTE
# -----------------------------------

smote = SMOTE(
    random_state=42
)

X_smote, y_smote = smote.fit_resample(

    X_final,
    y

)


# -----------------------------------
# FINAL MODEL
# -----------------------------------

model = RandomForestClassifier(

                n_estimators=200,
                max_depth=12,
                random_state=42

            )


model.fit(

    X_smote,
    y_smote

)


# -----------------------------------
# SAVE EVERYTHING
# -----------------------------------

joblib.dump(

    model,
    "random_forest_model.pkl"

)

joblib.dump(

    scaler,
    "scaler.pkl"

)

joblib.dump(

    encoder,
    "stock_encoder.pkl"

)


print(
    "Model saved successfully."
)