import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="Asset Comparison", layout="wide")
st.title("📊 Asset Comparison – 7-Day Predictions")

# -----------------------------
# PARAMETERS
# -----------------------------
SEQUENCE_LEN = 60
FORECAST_DAYS = 7
EPOCHS = 10

# Assets
stocks = ["AAPL", "MSFT", "GOOGL"]
cryptos = ["BTC-USD", "ETH-USD", "BNB-USD"]

# Select category
category = st.selectbox("Select Category", ["Stock", "Crypto"])
compare_assets = stocks if category=="Stock" else cryptos

# -----------------------------
# FUNCTIONS
# -----------------------------
def prepare_data(symbol):
    df = yf.download(symbol, period="5y", progress=False)
    df = df[['Close']].reset_index()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    X = []
    for i in range(SEQUENCE_LEN, len(scaled) - FORECAST_DAYS):
        X.append(scaled[i-SEQUENCE_LEN:i])
    return np.array(X), df, scaler

def train_model(X):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dense(FORECAST_DAYS)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# -----------------------------
# COMPARE BUTTON
# -----------------------------
if st.button(f"Compare All {category}s"):
    st.subheader(f"📈 Comparing {category}s – Next 7 Days Prediction")

    predicted_growth = {}
    fig, ax = plt.subplots(figsize=(12, 6))

    for sym in compare_assets:
        # Prepare data
        X, df, scaler = prepare_data(sym)
        model = train_model(X)
        model.fit(X, X[:, -FORECAST_DAYS:].reshape(len(X), FORECAST_DAYS),
                  epochs=EPOCHS, batch_size=32, verbose=0)

        # Predict next 7 days
        last_seq = X[-1].reshape(1, SEQUENCE_LEN, 1)
        future_scaled = model.predict(last_seq, verbose=0)[0]
        future_prices = scaler.inverse_transform(future_scaled.reshape(-1, 1)).flatten()

        # Calculate predicted growth % as float
        last_price = float(df['Close'].iloc[-1])
        growth = ((future_prices[-1] - last_price) / last_price) * 100
        predicted_growth[sym] = float(growth)  # ✅ force float

        # Prepare future dataframe for plotting
        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=FORECAST_DAYS, freq="B")
        future_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_prices})

        # Plot historical last year
        ax.plot(df['Date'].tail(252), df['Close'].tail(252), label=f"{sym} Historical")
        # Plot predicted next 7 days
        ax.plot(future_df["Date"], future_df["Predicted_Close"], linestyle="--", marker="o", label=f"{sym} Predicted")

    # Prediction start vertical line
    ax.axvline(x=df["Date"].iloc[-1], linestyle=":", linewidth=2, color="black", label="Prediction Start")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"{category}s: Historical + 7-Day Predictions")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # -----------------------------
    # Growth table (float-safe)
    # -----------------------------
    growth_df = pd.DataFrame(list(predicted_growth.items()), columns=["Asset", "Predicted 7-Day Growth (%)"])
    st.subheader("📋 Predicted 7-Day Growth Table")
    st.dataframe(growth_df.round(2))  # ✅ avoids formatting errors

    # -----------------------------
    # Bottom actions: Go Back + AI Suggestion
    # -----------------------------
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("⬅ Go Back"):
            st.switch_page("predict.py")  # Replace with your main page

    with col2:
        best_asset = max(predicted_growth, key=predicted_growth.get)
        st.success(f"💡 Ai Suggested Asset to Invest (Based on Predicted Growth): {best_asset} ({predicted_growth[best_asset]:.2f}%)")