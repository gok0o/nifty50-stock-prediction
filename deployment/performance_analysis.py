import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------
# LOAD DATA
# -----------------------------------

file_path = "predictions_log.csv"

df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

print("Total rows:", len(df))


# -----------------------------------
# DROP INCOMPLETE ROWS
# -----------------------------------

df = df.dropna(subset=["Prediction", "Actual"])

print("Valid rows (after drop):", len(df))


# -----------------------------------
# BASIC METRICS
# -----------------------------------

y_true = df["Actual"]
y_pred = df["Prediction"]

accuracy = accuracy_score(y_true, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# -----------------------------------
# CONFIDENCE ANALYSIS
# -----------------------------------

print("\n" + "="*50)
print("CONFIDENCE ANALYSIS")
print("="*50)

df["Correct"] = (df["Prediction"] == df["Actual"]).astype(int)

high_conf = df[df["Confidence"] >= 0.60]
low_conf = df[df["Confidence"] < 0.60]

if len(high_conf) > 0:
    high_acc = high_conf["Correct"].mean()
    print(f"\nHigh confidence accuracy (>=0.60): {high_acc:.4f}")
    print("Samples:", len(high_conf))
else:
    print("\nNo high confidence predictions yet.")

low_acc = low_conf["Correct"].mean()
print(f"Low confidence accuracy (<0.60): {low_acc:.4f}")
print("Samples:", len(low_conf))


# -----------------------------------
# PREDICTION DISTRIBUTION
# -----------------------------------

print("\n" + "="*50)
print("PREDICTION DISTRIBUTION")
print("="*50)

print("\nPrediction counts:")
print(df["Prediction"].value_counts())

print("\nActual counts:")
print(df["Actual"].value_counts())


# -----------------------------------
# SIMPLE PROFIT SIMULATION
# -----------------------------------

print("\n" + "="*50)
print("PROFIT SIMULATION")
print("="*50)

# Assume:
# +1 profit if correct
# -1 loss if wrong

df["Profit"] = np.where(df["Correct"] == 1, 1, -1)

total_profit = df["Profit"].sum()
avg_profit = df["Profit"].mean()

print(f"\nTotal Profit (units): {total_profit}")
print(f"Average Profit per trade: {avg_profit:.4f}")


# -----------------------------------
# OPTIONAL: ONLY TRADE HIGH CONFIDENCE
# -----------------------------------

high_conf_trades = df[df["Confidence"] >= 0.60]

if len(high_conf_trades) > 0:

    high_conf_trades["Profit"] = np.where(
        high_conf_trades["Prediction"] == high_conf_trades["Actual"], 1, -1
    )

    total_profit_hc = high_conf_trades["Profit"].sum()
    avg_profit_hc = high_conf_trades["Profit"].mean()

    print("\n--- HIGH CONFIDENCE STRATEGY ---")
    print(f"Trades taken: {len(high_conf_trades)}")
    print(f"Total Profit: {total_profit_hc}")
    print(f"Avg Profit: {avg_profit_hc:.4f}")

else:
    print("\nNo high confidence trades yet.")