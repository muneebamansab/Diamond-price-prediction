import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")

# ---------- Diamond theme ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #050b1f 0%, #0b1e4a 55%, #123a6b 100%);
}
h1 {
    text-align: center;
    background: linear-gradient(90deg, #ffffff, #7fdbff, #b9a7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #a9d8f5;
    margin-bottom: 1.5rem;
}
div.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.6rem 1rem;
}
div.stButton > button:hover {
    box-shadow: 0 0 18px #38bdf8;
    color: white;
}
.price-card {
    text-align: center;
    padding: 1.5rem;
    margin-top: 1rem;
    border-radius: 18px;
    background: rgba(127, 219, 255, 0.10);
    border: 1px solid #7fdbff;
    box-shadow: 0 0 25px rgba(127, 219, 255, 0.35);
}
.price-label { color: #a9d8f5; font-size: 1rem; }
.price-value { color: #ffffff; font-size: 2.8rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("diamond_model.pkl")

model = load_model()

# ---------- Header ----------
st.title("💎 Diamond Price Predictor")
st.markdown('<p class="subtitle">✨ Machine Learning model trained on 53,000+ real diamonds ✨</p>',
            unsafe_allow_html=True)

# ---------- Inputs ----------
st.subheader("🔹 Enter diamond details")
col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat (weight)", min_value=0.2, max_value=5.0, value=0.7, step=0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)

with col2:
    color = st.selectbox("Color (J = worst, D = best)", ["J", "I", "H", "G", "F", "E", "D"], index=3)
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=3)

# ---------- Prediction ----------
if st.button("💎 Predict Price", use_container_width=True):
    row = pd.DataFrame([[carat, cut, color, clarity]],
                       columns=["carat", "cut", "color", "clarity"])
    price = model.predict(row)[0]
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated Price (USD)</div>
        <div class="price-value">${price:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
with st.expander("ℹ️ About this project"):
    st.write("""
    - **Problem:** Diamond price prediction (regression)
    - **Dataset:** Kaggle Diamonds, 53,940 rows
    - **Algorithms compared:** Ridge, Lasso, ExtraTrees, AdaBoost, HistGradientBoosting
    - **Evaluation:** 10-fold cross-validation (RMSE)
    - **Best model:** HistGradientBoosting (RMSE about 557)
    """)