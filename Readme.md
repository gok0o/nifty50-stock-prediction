# 📈 Stock Direction Prediction System

A machine learning system designed to predict whether a stock will close **UP** or **DOWN** for the trading day using stock-level, market-level, sector-level, and peer-correlation features.

This project was built using data from **50 Nifty stocks** over approximately **10 years of historical data** from the :contentReference[oaicite:0]{index=0} universe.

---

## 🎯 Problem Statement

Traditional price prediction is noisy and difficult in financial markets.

Instead of predicting exact price values, this project focuses on a more practical classification problem:

> Will a stock close **above its opening price (UP)** or **below its opening price (DOWN)**?

This makes the model more useful for:

- Intraday directional analysis
- Quantitative trading research
- Market behavior studies

---

## 🧠 Approach

The project started with multiple model experiments:

### Deep Learning Experiments
- :contentReference[oaicite:1]{index=1}
- :contentReference[oaicite:2]{index=2}
- Transformer models

Although sequence models were tested, tree-based models produced better performance and more stable predictions on this dataset.

### Final Production Model
- :contentReference[oaicite:3]{index=3}

LightGBM was selected as the final production model because of:

- Better generalization
- Faster training
- Strong feature importance interpretability
- Better performance on tabular market data

---

## 📊 Feature Engineering

The model uses a combination of stock-specific and market-wide features.

### Stock Features
- Previous Day Return
- RSI (Relative Strength Index)
- Gap (Opening Gap)
- Relative Volume
- ATR (Average True Range)
- Previous Candle Range
- Previous Candle Body Strength
- Distance from Moving Average

### Market Features
- :contentReference[oaicite:4]{index=4} previous return
- :contentReference[oaicite:5]{index=5} return (market fear / volatility)

### Sector Features
- Sector Index Return
- Previous Sector Movement

### Cross-Stock Features
- Peer Stock Return

Example:

- :contentReference[oaicite:6]{index=6} ↔ :contentReference[oaicite:7]{index=7}
- :contentReference[oaicite:8]{index=8} ↔ :contentReference[oaicite:9]{index=9}

This helps the model capture sector rotation and inter-stock relationships.

---

## ⚙️ Model Calibration

Initial predictions showed a **bearish bias** using the default classification threshold:

Default decision rule:

:contentReference[oaicite:10]{index=10}

To improve class balance, probability threshold calibration was performed.

Final production threshold:

:contentReference[oaicite:11]{index=11}

This improved directional balance between bullish and bearish predictions.

---

## 📈 Model Performance

### Historical Validation

| Model | Accuracy |
|------|------|
| :contentReference[oaicite:12]{index=12} | ~65.9% |
| :contentReference[oaicite:13]{index=13} | ~65.7% |
| :contentReference[oaicite:14]{index=14} | ~65.2% |

Final selected model:

**:contentReference[oaicite:15]{index=15}**

---

## 📅 Recent 10-Day Market Testing

A recent market simulation was performed across multiple stocks:

- Total Predictions: **490**
- Threshold Used: **42%**
- Accuracy: **~50–53%**

This test reflects real-world market uncertainty and provides a more realistic evaluation.

---

## 📂 Project Structure

```bash
stock_prediction/
│
├── scripts/
│   ├── feature_engineering.py
│   ├── test_models.py
│   ├── save_lightgbm_model.py
│
├── deployment/
│   ├── live_predict.py
│   ├── last_10_day_prediction.py
│
├── models/
│   ├── lightgbm_model.pkl
│   ├── scaler.pkl
│   ├── stock_encoder.pkl
│
├── data/
│   ├── nifty_data.csv
│   ├── nifty_index.csv
│   ├── nifty_processed.csv
│
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

Potential next steps:

- News sentiment analysis
- Options chain data
- Global market correlations
- Earnings calendar effects
- Macro economic indicators

---

## 📌 Key Learnings

This project demonstrated:

- Financial feature engineering
- Model selection and comparison
- Class imbalance handling
- Probability threshold calibration
- Production deployment of ML models

---

## 👨‍💻 Author

Built as part of quantitative finance and machine learning research.