# pip install pandas numpy matplotlib scikit-learn tensorflow streamlit altair

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
from datetime import timedelta

# === Load Dataset ===
df = pd.read_csv("stock_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# === Settings ===
look_back = 60
forecast_horizon = 5
features = ['Close_Price', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR_14', 'Sentiment_Score', 'Price_Momentum']
tickers = df['Ticker'].unique()

# === Streamlit App ===
st.title("LSTM Stock Forecast: Forecasting, Risk, Scenario & Simulation")

selected_ticker = st.selectbox("Select a Ticker", sorted(tickers))
future_days = st.slider("Forecast future days (beyond 2024):", 5, 200, 30)
investment_amount = st.number_input("Enter Investment Amount (Optional for Risk Estimation):", min_value=0, value=10000)

os.makedirs("saved_models", exist_ok=True)
eval_results = []

def get_market_scenario(forecast_prices):
    if forecast_prices[-1] > forecast_prices[0] * 1.05:
        return "📈 Bullish"
    elif forecast_prices[-1] < forecast_prices[0] * 0.95:
        return "📉 Bearish"
    else:
        return "🔄 Sideways"

def assess_risk_smart(forecast_prices):
    volatility = np.std(forecast_prices) / np.mean(forecast_prices)
    peak = forecast_prices[0]
    max_drawdown = 0
    for price in forecast_prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Fallbacks if drawdown is zero
    if max_drawdown == 0 and volatility > 0.02:
        max_drawdown = 0.03

    if max_drawdown > 0.10:
        return "High Risk", "red", max_drawdown, volatility
    elif volatility > 0.05:
        return "High Risk", "red", max_drawdown, volatility
    elif volatility > 0.02:
        return "Medium Risk", "orange", max_drawdown, volatility
    else:
        return "Low Risk", "green", max_drawdown, volatility

def colored_badge(text, color):
    return f"<span style='color:white;background-color:{color};padding:5px;border-radius:5px;'>{text}</span>"

for ticker in tickers:
    df_ticker = df[df['Ticker'] == ticker].copy().sort_values('Date')
    data = df_ticker[features].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data) - forecast_horizon):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i:i + forecast_horizon, 0])
    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model_path = f"saved_models/lstm_{ticker}.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(forecast_horizon)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        model.save(model_path)

    y_pred = model.predict(X_test)
    num_features = scaled_data.shape[1]
    y_pred_padded = np.hstack([y_pred, np.zeros((y_pred.shape[0], num_features - y_pred.shape[1]))])
    y_test_padded = np.hstack([y_test, np.zeros((y_test.shape[0], num_features - y_test.shape[1]))])
    y_pred_rescaled = scaler.inverse_transform(y_pred_padded)[:, 0]
    y_test_rescaled = scaler.inverse_transform(y_test_padded)[:, 0]

    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    eval_results.append({"Ticker": ticker, "MAE": mae, "RMSE": rmse, "R2": r2})

    if ticker == selected_ticker:
        st.subheader(f"Actual vs Predicted: {ticker}")
        st.line_chart(pd.DataFrame({"Actual": y_test_rescaled, "Predicted": y_pred_rescaled}))

        # === Future Forecast ===
        last_sequence = scaled_data[-look_back:]
        future_input = last_sequence.reshape(1, look_back, num_features)
        future_predictions = []

        for _ in range(future_days):
            future_step = model.predict(future_input)[0][0]
            next_input = np.append(future_input[:, 1:, :], [[[future_step] + [0] * (num_features - 1)]], axis=1)
            future_input = next_input
            future_predictions.append(future_step)

        future_preds = scaler.inverse_transform(
            np.hstack([np.array(future_predictions).reshape(-1, 1), np.zeros((future_days, num_features - 1))])
        )[:, 0]

        last_date = df_ticker['Date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
        future_df = pd.DataFrame({"Date": future_dates, "Forecasted_Close": future_preds})

        # === Combined Chart using Altair
        combined_df = pd.concat([
            pd.DataFrame({'Date': df_ticker['Date'], 'Price': df_ticker['Close_Price'], 'Type': 'Historical'}),
            pd.DataFrame({'Date': future_dates, 'Price': future_preds, 'Type': 'Forecasted'})
        ])

        st.subheader("Combined Historical + Forecast")
        chart = alt.Chart(combined_df).mark_line().encode(
            x='Date:T', y='Price:Q', color='Type:N'
        ).properties(width=800, height=400)
        st.altair_chart(chart, use_container_width=True)

        # === Market Scenario & Risk
        market_scenario = get_market_scenario(future_preds)
        risk_level, risk_color, max_drawdown, volatility = assess_risk_smart(future_preds)

        st.subheader("Market Scenario")
        st.markdown(f"**{ticker}** is projected as: **{market_scenario}**")

        st.subheader("Investment Risk")
        st.markdown(colored_badge(risk_level, risk_color), unsafe_allow_html=True)

        if investment_amount > 0:
            potential_loss = investment_amount * max_drawdown
            st.subheader("Potential Worst-Case Loss")
            st.markdown(
                f"<div style='background-color:#0e1117; padding:12px; border-radius:6px; color:#ffffff;'>"
                f"If you invest <strong>${investment_amount:,.0f}</strong>, the worst-case potential loss could be approximately "
                f"<strong>${potential_loss:,.2f}</strong> based on forecasted drawdowns."
                f"</div>",
                unsafe_allow_html=True
            )

        st.subheader("Insight Summary")
        summary = (
            f"**{ticker}** is expected to show a {market_scenario.lower()} trend over **{future_days} days**.\n\n"
            f"Risk is **{risk_level}**, and a **${investment_amount:,.0f}** investment might face a potential loss of "
            f"**${potential_loss:,.2f}**."
        )
        st.markdown(summary)

        st.subheader("What-If Scenario Simulation")
        drop_pct = st.slider("Simulate price drop (%)", 0, 50, 5)
        sim_loss = investment_amount * max_drawdown * (1 + drop_pct / 100)
        st.info(f"After a simulated {drop_pct}% drop, worst-case loss = ${sim_loss:,.2f}")

        st.caption(f"Volatility: {volatility:.4f}, Max Drawdown: {max_drawdown:.4f}")

        st.download_button("📥 Download Forecast CSV",
                           data=future_df.to_csv(index=False).encode('utf-8'),
                           file_name=f'{ticker}_forecast.csv',
                           mime='text/csv')

# === Final Evaluation Summary
eval_df = pd.DataFrame(eval_results)
eval_df.to_csv("saved_models/evaluation_report.csv", index=False)

st.subheader("Model Performance Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Avg MAE", f"{eval_df['MAE'].mean():.2f}")
col2.metric("Avg RMSE", f"{eval_df['RMSE'].mean():.2f}")
col3.metric("Avg R²", f"{eval_df['R2'].mean():.2f}")

st.download_button("📤 Download Evaluation Report (CSV)",
                   data=eval_df.to_csv(index=False).encode('utf-8'),
                   file_name="evaluation_report.csv",
                   mime="text/csv")

st.success("✅ Forecasting, Scenario Analysis, Risk, and Insight Summary Ready!")
