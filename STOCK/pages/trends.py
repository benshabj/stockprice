import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Asset Comparison", layout="wide")

# ---------------- HIDE SIDEBAR ----------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CHECK SESSION ----------------
if "asset_type" not in st.session_state or "asset" not in st.session_state:
    st.warning("Please start from Home page")
    st.stop()

category = st.session_state["asset_type"]   # Stock / Crypto
base_asset = st.session_state["asset"]

st.title(" Asset Comparison – 7-Day Predictions")
st.caption(f"Base Asset: **{base_asset}** ({category})")

# ---------------- PARAMETERS ----------------
SEQUENCE_LEN = 60
FORECAST_DAYS = 7
EPOCHS = 10

# ---------------- ASSET LIST ----------------
stocks = ["AAPL", "MSFT", "GOOGL"]
cryptos = ["BTC-USD", "ETH-USD", "BNB-USD"]

compare_assets = stocks if category == "Stock" else cryptos

if base_asset not in compare_assets:
    compare_assets.insert(0, base_asset)

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

# ---------------- AUTO COMPARISON ----------------
st.subheader(" Comparing Assets – Next 7 Days Prediction")

predicted_growth = {}
fig, ax = plt.subplots(figsize=(12, 6))

for sym in compare_assets:

    X, y, df, scaler = prepare_data(sym)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]

    model = build_model(X_train)
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=32,
        verbose=0
    )

    # -------- FUTURE PREDICTION --------
    last_seq = X[-1].reshape(1, SEQUENCE_LEN, 1)
    future_scaled = model.predict(last_seq, verbose=0)[0]

    future_prices = scaler.inverse_transform(
        future_scaled.reshape(-1, 1)
    ).flatten()

    last_price = float(df["Close"].iloc[-1])
    growth = ((future_prices[-1] - last_price) / last_price) * 100
    predicted_growth[sym] = float(growth)

    future_dates = pd.date_range(
        start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="B"
    )

    # -------- GRAPH (LIKE OLD VERSION) --------
    ax.plot(
        df["Date"].tail(252),
        df["Close"].tail(252),
        linewidth=2,
        label=f"{sym} Historical"
    )

    ax.plot(
        future_dates,
        future_prices,
        linestyle="--",
        marker="o",
        label=f"{sym} Predicted"
    )

# Prediction start line
ax.axvline(
    df["Date"].iloc[-1],
    linestyle=":",
    linewidth=2,
    color="black",
    label="Prediction Start"
)

ax.set_title("Assets: Historical + 7-Day Prediction Comparison")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ---------------- GROWTH TABLE ----------------
st.subheader(" Predicted 7-Day Growth (%)")

growth_df = (
    pd.DataFrame(predicted_growth.items(),
                 columns=["Asset", "Predicted Growth (%)"])
    .sort_values("Predicted Growth (%)", ascending=False)
)

st.dataframe(growth_df.round(2), use_container_width=True)

# ---------------- AI SUGGESTION ----------------
best_asset = growth_df.iloc[0]["Asset"]
best_value = growth_df.iloc[0]["Predicted Growth (%)"]

st.success(
    f"💡 AI Suggested Asset (Based on Prediction): "
    f"**{best_asset}** ({best_value:.2f}%)"
)

# ---------------- NAVIGATION ----------------
col1, col2 = st.columns(2)


with col1:
    
    if st.button("⬅ Back to Prediction"):
        st.switch_page("pages/predict.py")

with col2:
    if st.button(" Home"):
        st.switch_page("app.py")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        color: gray;
        font-size: 13px;
    }
    </style>

    <div class="footer">
        ⚠️ Educational purpose only. Not financial advice.
    </div>
    """,
    unsafe_allow_html=True
)
# trends.py