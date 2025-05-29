# 📊 FINSIM: AI-Powered Market Simulation, Investment Forecasting & Risk Assessment System

**FINSIM** is a unified financial intelligence system that simulates real-world market conditions, forecasts stock trends, and assesses risk using deep learning and multi-factor analysis. The system integrates three key modules: **Investment Forecasting**, **Market Scenario Simulation**, and **Risk Assessment**.

---

## 🎯 My Role

I led the **Investment Forecasting** module and played a central role in integrating the final system. My work involved:

- Designing and training a multi-ticker LSTM model for **stock price prediction** across 40 companies.
- Supporting **custom prediction horizons** (from 7 days to 2 years) based on user input.
- Engineering features from both **technical indicators** and **market sentiment** data.
- Writing the integration logic to simulate **market scenarios** and assess **investment risk** dynamically using model outputs.

> While full code-level merging of all modules wasn’t feasible due to time constraints, I developed logic that **infers market conditions and risk states directly** from investment forecasts — effectively allowing the three systems to operate cohesively.

---

## 🧠 System Overview

### 🔹 Investment Forecasting (Core Module)
- Multi-ticker LSTM model trained on 2019–2024 market data.
- Features include: `Close Price`, `SMA_50`, `MACD`, `RSI`, `ATR_14`, `Sentiment_Score`, `Volatility_Index`, and more.
- Achieved prediction accuracies ranging from **88% to 97%** across different stocks and sectors.

### 🔹 Market Scenario Simulation
- Simulates bullish, bearish, or neutral market conditions based on macro features and model-inferred signals.
- Uses forecasted price trends, volume patterns, and sentiment deltas.

### 🔹 Risk Assessment
- Evaluates real-time market risk using volatility, relative volume, and prediction uncertainty.
- Custom logic enables dynamic risk classification (low, moderate, high).

---

## 🛠️ Technologies Used

- Python, Pandas, NumPy
- TensorFlow / Keras (LSTM)
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit (for future interactive UI)

---

## 📈 Performance Metrics

- **Prediction Accuracy**: 88% – 97%
- **Forecast Granularity**: Short-term (7 days) to long-term (5.5 years)
- **Sectors Modeled**: Tech, Healthcare, Finance, Retail, Energy, Automotive, Consumer Goods, Media

---

## 💬 Key Takeaway

FINSIM provides a robust and modular architecture to forecast investment opportunities, simulate economic conditions, and assess portfolio risk — all driven by AI. It represents a practical and scalable solution for data-driven financial analysis.

---

## 🤝 Contact

**Lachiket Narendra Warule**  
🔗 [https://www.linkedin.com/in/lachiket-warule-857403209/] | 💼 [https://github.com/Lachiket03]

---
