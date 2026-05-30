import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------------
# LOAD DATA
# -----------------------------------

X = np.load("X_lstm.npy")
y = np.load("y_lstm.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)


# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# -----------------------------------
# CLASS WEIGHTS
# -----------------------------------

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print("Class Weights:", class_weights)

# -----------------------------------
# BUILD MODEL
# -----------------------------------

model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


# -----------------------------------
# COMPILE
# -----------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------------
# EARLY STOPPING
# -----------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# -----------------------------------
# TRAIN
# -----------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=128,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)


# -----------------------------------
# PREDICT
# -----------------------------------

y_prob = model.predict(X_test)

y_pred = (
    y_prob > 0.5
).astype(int)


# -----------------------------------
# EVALUATE
# -----------------------------------

print("\n==================================================")
print("LSTM RESULTS")
print("==================================================")

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, y_pred),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(y_test, y_pred),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(y_test, y_pred),
        4
    )
)

print(
    "F1 Score:",
    round(
        f1_score(y_test, y_pred),
        4
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# -----------------------------------
# SAVE MODEL
# -----------------------------------

model.save(
    "lstm_model.h5"
)

print("\nLSTM model saved.")