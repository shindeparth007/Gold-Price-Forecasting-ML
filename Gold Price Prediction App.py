
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta

st.set_page_config(page_title="Gold Price Predictor (Fixed CSV)", layout="wide")
st.title("🪙 Gold Price Predictor (Fixed CSV only)")
st.caption("Data source is locked to **gold_price_predictions_final.csv** — no other files allowed.")

CSV_PATH = "gold_price_predictions_final.csv"  # Local file next to this script

def friendly_error_box(e: Exception):
    st.error("Something went wrong. See details below.")
    with st.expander("Show technical details"):
        st.exception(e)

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Detect date col
    date_cols = [c for c in df.columns if c.lower() in ["date", "day", "timestamp"]]
    if not date_cols:
        df = df.copy()
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
        date_col = "Date"
    else:
        date_col = date_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "Date"})

    # Detect price col
    candidates = ["Actual_Price_INR", "Gold_Price", "Gold Price", "Price", "Close", "Close_Price", "Close Price"]
    price_col = next((c for c in candidates if c in df.columns), None)
    if price_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 1:
            price_col = numeric_cols[0]
        else:
            raise ValueError("Could not find a price column. Expected one of: " + ", ".join(candidates))

    df = df[["Date", price_col]].rename(columns={price_col: "Price"})
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"]).reset_index(drop=True)

    # Daily freq & fill
    df = df.set_index("Date").asfreq("D")
    df["Price"] = df["Price"].interpolate(method="time").bfill().ffill()
    df = df.reset_index()
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Lags
    df["lag_1"] = df["Price"].shift(1)
    df["lag_7"] = df["Price"].shift(7)
    df["lag_14"] = df["Price"].shift(14)
    df["lag_30"] = df["Price"].shift(30)
    # Rollings
    df["roll_7"] = df["Price"].rolling(7).mean()
    df["roll_14"] = df["Price"].rolling(14).mean()
    df["roll_30"] = df["Price"].rolling(30).mean()
    # Calendar
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df = df.dropna().reset_index(drop=True)
    return df

def train_model(df_feat: pd.DataFrame):
    if len(df_feat) < 50:
        st.warning(f"Only {len(df_feat)} rows after feature engineering; results may be unstable.")
    split_idx = int(len(df_feat) * 0.8)
    if split_idx <= 5 or (len(df_feat) - split_idx) < 5:
        raise ValueError("Not enough rows for a train/test split. Add more data.")

    train = df_feat.iloc[:split_idx]
    test  = df_feat.iloc[split_idx:]

    feature_cols = [c for c in df_feat.columns if c not in ["Date", "Price"]]
    X_train = train[feature_cols]
    y_train = train["Price"].astype(float)
    X_test  = test[feature_cols]
    y_test  = test["Price"].astype(float)

    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return model, metrics, feature_cols, train, test

def predict_at_date(model, feature_cols, history_df: pd.DataFrame, target_date: pd.Timestamp) -> float:
    """
    Use trained feature order:
      - For in-sample: take the exact row's engineered features
      - For future: recursively roll forward using previous day's engineered features,
        but update calendar fields to the current day.
    """
    df = history_df.copy().set_index("Date")
    last_date = df.index.max()

    # In-sample
    if target_date in df.index and target_date <= last_date:
        row = df.loc[[target_date]].copy()
        X = row.drop(columns=["Price"], errors="ignore")
        # Ensure correct order and drop non-feature cols if any leaked
        X = X.reindex(columns=feature_cols)
        X = X.astype(float)
        return float(model.predict(X.values)[0])

    # Future recursive
    cur_date = last_date + timedelta(days=1)
    df_future = df.copy()

    while cur_date <= target_date:
        # Build a helper table with features including the reference day values
        hist = df_future.copy().reset_index()
        # Recompute engineered columns for hist
        hist["lag_1"] = hist["Price"].shift(1)
        hist["lag_7"] = hist["Price"].shift(7)
        hist["lag_14"] = hist["Price"].shift(14)
        hist["lag_30"] = hist["Price"].shift(30)
        hist["roll_7"] = hist["Price"].rolling(7).mean()
        hist["roll_14"] = hist["Price"].rolling(14).mean()
        hist["roll_30"] = hist["Price"].rolling(30).mean()
        hist["dayofweek"] = hist["Date"].dt.dayofweek
        hist["month"] = hist["Date"].dt.month
        hist["day"] = hist["Date"].dt.day

        ref_day = cur_date - timedelta(days=1)
        ref_rows = hist[hist["Date"] == ref_day]
        if ref_rows.empty:
            raise ValueError("Insufficient history to compute features for recursive forecast.")

        # Start from previous day's engineered features, exclude Date/Price
        last_row = ref_rows.iloc[-1].to_dict()
        feat = {col: last_row[col] for col in feature_cols if col in last_row}

        # Override calendar features for the *current* day
        if "dayofweek" in feat: feat["dayofweek"] = cur_date.weekday()
        if "month" in feat:     feat["month"]     = cur_date.month
        if "day" in feat:       feat["day"]       = cur_date.day

        X = pd.DataFrame([feat], columns=feature_cols)
        X = X.astype(float)

        yhat = float(model.predict(X.values)[0])
        df_future.loc[cur_date, "Price"] = yhat
        cur_date += timedelta(days=1)

    if target_date not in df_future.index:
        raise ValueError("Target date not produced during recursive forecast.")
    return float(df_future.loc[target_date, "Price"])

# ---------------- Main ----------------
try:
    raw_df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load the fixed CSV `{CSV_PATH}`.")
    with st.expander("Show technical details"):
        st.exception(e)
    st.stop()

st.success(f"Loaded {len(raw_df):,} rows from the fixed CSV.")
with st.expander("Peek at data (tail)"):
    st.dataframe(raw_df.tail(20), use_container_width=True)

# Engineer features
feat_df = make_features(raw_df)

# Train
try:
    model, metrics, feature_cols, train, test = train_model(feat_df)
except Exception as e:
    friendly_error_box(e)
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{metrics['MAE']:.2f}")
c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
c3.metric("R²", f"{metrics['R2']:.3f}")

st.markdown("---")
st.subheader("🔮 Forecast")

min_date = raw_df["Date"].min().date()
max_date = raw_df["Date"].max().date()
default_future = max_date + timedelta(days=7)

target = st.date_input(
    "Pick a date to predict",
    value=default_future,
    min_value=min_date,
    max_value=max_date + timedelta(days=365)
)

if st.button("Predict for selected date"):
    tstamp = pd.to_datetime(target)
    try:
        yhat = predict_at_date(model, feature_cols, feat_df, tstamp)
        st.success(f"Predicted Price on {tstamp.date().isoformat()}: **{yhat:,.2f}**")
    except Exception as e:
        friendly_error_box(e)

st.caption("Fixes: (1) Drops 'Date' from features, (2) Enforces training feature order during prediction, (3) Casts features to float to avoid Timestamp errors.")
