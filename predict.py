import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stock Prediction", layout="wide")

# ---------------- HIDE SIDEBAR ----------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CHECK SESSION ----------------
if "asset" not in st.session_state:
    st.warning("Please select an asset first")
    st.stop()

asset = st.session_state["asset"]

st.title(f"📈 {asset} – Next 7 Days Prediction")

# ---------------- PARAMETERS ----------------
SEQUENCE_LEN = 60
FORECAST_DAYS = 7
EPOCHS = 10

# ---------------- FUNCTIONS ----------------
def prepare_data(symbol):
    df = yf.download(symbol, period="5y", progress=False)

    df = df[['Close']].reset_index()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])

    X, y = [], []
    for i in range(SEQUENCE_LEN, len(scaled) - FORECAST_DAYS):
        X.append(scaled[i-SEQUENCE_LEN:i])
        y.append(scaled[i:i+FORECAST_DAYS])

    return np.array(X), np.array(y), df, scaler


def build_model(X):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dense(FORECAST_DAYS)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------------- AUTO PREDICT ----------------
with st.spinner("Training model and predicting..."):

    X, y, df, scaler = prepare_data(asset)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model(X_train)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0)

    preds = model.predict(X_test, verbose=0)

    mae = mean_absolute_error(y_test.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), preds.flatten()))

    # -------- FUTURE PREDICTION --------
    last_seq = X[-1].reshape(1, SEQUENCE_LEN, 1)
    future_scaled = model.predict(last_seq, verbose=0)[0]

    future_prices = scaler.inverse_transform(
        future_scaled.reshape(-1, 1)
    ).flatten()

    # -------- FIX FOR AMBIGUOUS SERIES ERROR --------
    last_price = df['Close'].iloc[-1].item()
    future_last_price = future_prices[-1].item()

    trend = "UP 📈" if future_last_price > last_price else "DOWN 📉"

    # -------- FUTURE DATES --------
    future_dates = pd.date_range(
        start=df['Date'].iloc[-1] + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="B"
    )

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Close": future_prices
    })

# ---------------- OUTPUT ----------------
st.subheader("📋 Predicted Prices")
st.dataframe(future_df.round(2), use_container_width=True)

st.subheader("📈 Prediction Chart")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Date'].tail(252), df['Close'].tail(252), label="Historical")
ax.plot(
    future_df['Date'],
    future_df['Predicted Close'],
    marker="o",
    linestyle="--",
    label="Prediction"
)
ax.axvline(df['Date'].iloc[-1], linestyle=":", color="gray")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("📊 Accuracy")
col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.4f}")
col2.metric("RMSE", f"{rmse:.4f}")

st.success(f"💡 Expected Trend: **{trend}**")

# ---------------- NAVIGATION ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("⬅ Go Back"):
        st.switch_page("app.py")

with col2:
    if st.button("📊 Market Comparison"):
        st.switch_page("pages/trends.py")

st.caption("⚠️ Educational purpose only. Not financial advice.")
