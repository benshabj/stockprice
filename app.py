import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stock & Crypto Predictor", layout="wide")

# ---------------- HIDE SIDEBAR ----------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<h1 style="text-align:center;">WELCOME</h1>
<h3 style="text-align:center; color:gray;">
Stock & Crypto Prediction
</h3>
""", unsafe_allow_html=True)

st.write("")

# ---------------- CENTER UI ----------------
left, center, right = st.columns([2,3,2])

with center:
    col1, col2 = st.columns(2)

    with col1:
        asset_type = st.selectbox("Select Type", ["Stock", "Crypto"])

    with col2:
        if asset_type == "Stock":
            asset = st.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL"])
        else:
            asset = st.selectbox("Select Crypto", ["BTC-USD", "ETH-USD", "BNB-USD"])

    st.write("")

    if st.button("Predict"):
        # 🔑 Save selection globally
        st.session_state["asset_type"] = asset_type
        st.session_state["asset"] = asset

        # Go to prediction page
        st.switch_page("predict.py")
