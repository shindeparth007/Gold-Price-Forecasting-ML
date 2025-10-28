import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Gold Price Forecasting (India)", layout="wide")
st.title("🏆 Gold Price Forecasting (Indian Market)")

st.markdown("""
This app predicts **Indian Gold Prices (INR)** using a preloaded dataset.  
Just select any date — past or future — to get the estimated gold price prediction.
""")

# -------------------- LOAD FIXED DATASET --------------------
DATA_PATH = "gold_price_predictions_final.csv"  # ✅ Keep this file in the same folder as app.py

@st.cache_data
def load_data():
    # Load CSV
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    # ✅ Fixed column names from your file
    date_col = "Date"
    target_col = "Actual_Price_INR"

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Clean and sort
    df = df[[date_col, target_col]].dropna().copy()
    df = df.sort_values(by=date_col)
    df.rename(columns={date_col: "Date", target_col: "GoldPrice"}, inplace=True)

    # Add time and seasonality features
    df["t"] = (df["Date"] - df["Date"].min()).dt.days
    df["month"] = df["Date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df

try:
    df = load_data()
    st.success("✅ Dataset loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

st.caption(f"Data range: {df['Date'].min().date()} → {df['Date'].max().date()} | Records: {len(df)}")

# -------------------- TRAIN MODEL AUTOMATICALLY --------------------
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[["t", "month_sin", "month_cos"]]
y_train = train_df["GoldPrice"]
X_test = test_df[["t", "month_sin", "month_cos"]]
y_test = test_df["GoldPrice"]

# Polynomial + Ridge Regression
degree = 2
alpha = 1.0
poly = PolynomialFeatures(degree=degree, include_bias=False)
pre = ColumnTransformer([
    ("poly_t", poly, ["t"]),
    ("season", "passthrough", ["month_sin", "month_cos"])
])

model = Pipeline([
    ("pre", pre),
    ("ridge", Ridge(alpha=alpha, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Mean Squared Error (↓)", f"{mse:,.2f}")
with col2:
    st.metric("R² Score (↑)", f"{r2:.4f}")

# -------------------- VISUALIZATION --------------------
fig, ax = plt.subplots(figsize=(8,3))
ax.plot(test_df["Date"], y_test, label="Actual", color="black")
ax.plot(test_df["Date"], y_pred, label="Predicted", color="gold")
ax.set_xlabel("Date"); ax.set_ylabel("Gold Price (INR)")
ax.set_title("Actual vs Predicted Gold Prices (Test Period)")
ax.legend(); ax.grid(True, alpha=0.3)
st.pyplot(fig)

# -------------------- DATE INPUT FOR PREDICTION --------------------
st.subheader("🗓️ Predict Gold Price for a Given Date")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
st.caption(f"Training window: {min_date} → {max_date}. You can pick any past or future date.")

user_date = st.date_input("Select a date for prediction", value=max_date)

pred_date = pd.to_datetime(user_date)
t_val = (pred_date - df["Date"].min()).days
month = pred_date.month

input_features = pd.DataFrame({
    "t": [t_val],
    "month_sin": [np.sin(2 * np.pi * month / 12)],
    "month_cos": [np.cos(2 * np.pi * month / 12)]
})

predicted_price = float(model.predict(input_features)[0])
st.success(f"💰 Predicted Gold Price for {user_date}: ₹{predicted_price:,.2f}")

# -------------------- OPTIONAL: FORECAST TREND --------------------
with st.expander("📉 View Forecast Trend for Next 6 Months"):
    future_days = 180
    future_dates = pd.date_range(df["Date"].max(), periods=future_days)
    t_future = (future_dates - df["Date"].min()).days
    months = future_dates.month

    fut_df = pd.DataFrame({
        "t": t_future,
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12)
    })
    fut_df["Predicted"] = model.predict(fut_df)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["Date"], df["GoldPrice"], label="Historical", color="black")
    ax.plot(future_dates, fut_df["Predicted"], label="Forecast (Next 6 months)", color="gold")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.markdown("---")
st.markdown("🧠 *Model trained on Indian gold price trends using polynomial regression and seasonality patterns.*")
