import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.model_selection import (
    RandomizedSearchCV
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv(
    "nifty_processed.csv"
)


# -----------------------------------
# STOCK ENCODING
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
    "Prev_Target",
    "Gap",
    "Prev_Index_Return"

]

target_col = "Target"


# -----------------------------------
# TIME SPLIT
# -----------------------------------

split = int(
    len(df) * 0.8
)

train_df = df.iloc[:split]
test_df = df.iloc[split:]


# -----------------------------------
# SCALE
# -----------------------------------

scaler = StandardScaler()

X_train_tech = scaler.fit_transform(
    train_df[feature_cols]
)

X_test_tech = scaler.transform(
    test_df[feature_cols]
)


X_train = np.column_stack((

    X_train_tech,
    train_df["Stock_ID"].values

))

X_test = np.column_stack((

    X_test_tech,
    test_df["Stock_ID"].values

))


y_train = train_df[
    target_col
].values

y_test = test_df[
    target_col
].values


# -----------------------------------
# SMOTE
# -----------------------------------

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(

    X_train,
    y_train

)


# -----------------------------------
# XGBOOST SEARCH SPACE
# -----------------------------------

param_grid = {

    "n_estimators":
        [100, 200, 300],

    "learning_rate":
        [0.01, 0.05, 0.1],

    "max_depth":
        [4, 6, 8, 10],

    "subsample":
        [0.7, 0.8, 1.0],

    "colsample_bytree":
        [0.7, 0.8, 1.0]

}


model = XGBClassifier(

    random_state=42,
    eval_metric="logloss",
    verbosity=0

)


search = RandomizedSearchCV(

    estimator=model,

    param_distributions=param_grid,

    n_iter=15,

    scoring="f1",

    cv=3,

    verbose=2,

    n_jobs=-1,

    random_state=42

)


print(
    "Starting XGBoost tuning..."
)

search.fit(

    X_train_smote,
    y_train_smote

)


# -----------------------------------
# BEST MODEL
# -----------------------------------

best_model = search.best_estimator_

print(
    "\nBest Parameters:"
)

print(
    search.best_params_
)


# -----------------------------------
# FINAL TEST
# -----------------------------------

preds = best_model.predict(
    X_test
)

print(
    "\nFinal Classification Report:"
)

print(

    classification_report(
        y_test,
        preds
    )

)

print(
    "\nConfusion Matrix:"
)

print(

    confusion_matrix(
        y_test,
        preds
    )

)

print(
    "\nFinal F1:"
)

print(

    f1_score(
        y_test,
        preds
    )

)