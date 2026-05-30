import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

# Models
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def run_elite_testing():

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------

    try:

        df = pd.read_csv(
            "nifty_processed.csv"
        )

    except FileNotFoundError:

        print(
            "Error: Run feature_engineering.py first."
        )

        return


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
        "DayOfWeek",
        "Month"

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
    # SCALE FEATURES
    # -----------------------------------

    scaler = StandardScaler()

    X_train_tech = scaler.fit_transform(
        train_df[feature_cols]
    )

    X_test_tech = scaler.transform(
        test_df[feature_cols]
    )


    # Add Stock_ID
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
    # MODELS
    # -----------------------------------

    models = {

        "XGBoost":

            XGBClassifier(

                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                verbosity=0,
                random_state=42

            ),

        "LightGBM":

            LGBMClassifier(

                n_estimators=200,
                learning_rate=0.05,
                verbose=-1,
                random_state=42

            ),

        "Random Forest":

            RandomForestClassifier(

                n_estimators=200,
                max_depth=12,
                random_state=42

            ),

        "Extra Trees":

            ExtraTreesClassifier(

                n_estimators=200,
                random_state=42

            )
    }


    # -----------------------------------
    # TRAIN + EVALUATE
    # -----------------------------------

    results = []

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"TRAINING {model_name}")
        print("=" * 60)

        # Train
        model.fit(

            X_train_smote,
            y_train_smote

        )
        
        # Predict
        preds = model.predict(
            X_test
        )

        # Metrics
        accuracy = accuracy_score(
            y_test,
            preds
        )

        precision = precision_score(
            y_test,
            preds
        )

        recall = recall_score(
            y_test,
            preds
        )

        f1 = f1_score(
            y_test,
            preds
        )

        cm = confusion_matrix(
            y_test,
            preds
        )

        print(
            f"\nAccuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            "\nClassification Report:"
        )

        print(

            classification_report(
                y_test,
                preds
            )

        )

        print(
            "Confusion Matrix:"
        )

        print(
            cm
        )

        results.append({

            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1

        })
        if model_name == "LightGBM":

            importance_df = pd.DataFrame({

                "Feature": feature_cols + ["Stock_ID"],
                "Importance": model.feature_importances_

            })

            importance_df = importance_df.sort_values(

                "Importance",
                ascending=False

            )

            print("\n")
            print("=" * 50)
            print("LIGHTGBM FEATURE IMPORTANCE")
            print("=" * 50)

            print(
                importance_df
            )


    # -----------------------------------
    # FINAL RANKING
    # -----------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(

        by="F1",
        ascending=False

    )

    print("\n")
    print("=" * 60)
    print("FINAL MODEL RANKING")
    print("=" * 60)

    print(
        results_df
    )


if __name__ == "__main__":

    run_elite_testing()