
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from polygon_data import get_stock_data
from utils.model_selector import select_best_model
from utils.visuals import plot_predictions
from utils.explainable_ai import explain_decision

from models.lstm_model import train_lstm
from models.gru_model import train_gru
from models.mlp_model import train_mlp
from models.xgboost_model import train_xgb

st.set_page_config(layout="wide")
st.title("AI Trading Enterprise Suite with Auto-Training")

symbol = st.sidebar.text_input("ادخل رمز السهم", "AAPL").upper()
days = st.sidebar.slider("عدد الأيام للبيانات", 90, 720, 365)
future_days = st.sidebar.slider("عدد الأيام للتوقع", 1, 7, 3)
use_saved = st.sidebar.checkbox("تحميل النموذج المدرب مسبقًا إذا وجد")
run = st.sidebar.button("ابدأ التحليل")

if run:
    st.subheader("تحميل البيانات")
    df = get_stock_data(symbol, days=days)

    if df.empty or len(df) < 100:
        st.error("البيانات غير كافية.")
        st.stop()

    st.success(f"تم تحميل {len(df)} صف من البيانات")

    df["Returns"] = df["close"].pct_change()
    df.dropna(inplace=True)
    prices = df["close"].values.reshape(-1, 1)

    window = 30
    X, y = [], []
    for i in range(window, len(prices) - future_days):
        X.append(prices[i-window:i, 0])
        y.append(prices[i:i+future_days, 0])
    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model_path = f"models/saved/{symbol}_best_model.pkl"
    predictions = {}

    if use_saved and os.path.exists(model_path):
        st.success("تم تحميل النموذج المدرب مسبقًا.")
        predictions = joblib.load(model_path)
    else:
        st.warning("يتم الآن تدريب النماذج على البيانات المحدثة...")
        predictions["LSTM"] = train_lstm(X_train, y_train, X_test)
        predictions["GRU"] = train_gru(X_train, y_train, X_test)
        predictions["MLP"] = train_mlp(X_train, y_train, X_test)
        predictions["XGBoost"] = train_xgb(X_train, y_train, X_test)
        joblib.dump(predictions, model_path)
        st.success("تم حفظ النموذج الأفضل محليًا.")

    best_model, best_mae = select_best_model(y_test, predictions)
    st.success(f"النموذج الأفضل: {best_model} | MAE = {best_mae:.4f}")

    # التوصيات
    last_prediction = predictions[best_model][-1]
    forecast_dates = pd.date_range(start=df.index[-1], periods=future_days+1, freq="B")[1:]
    forecast_prices = pd.Series(last_prediction, index=forecast_dates)

    st.subheader("التوصيات المستقبلية:")
    recs = ["شراء" if p > prices[-1] else "بيع" for p in last_prediction]
    df_result = pd.DataFrame({"السعر المتوقع": last_prediction, "التوصية": recs}, index=forecast_dates)
    st.dataframe(df_result)

    st.subheader("الرسم البياني للتنبؤ:")
    plot_predictions(y_test[:, -1], predictions[best_model][:, -1], title=f"{best_model} توقعات")

    indicators = {"MAE": best_mae, "آخر سعر": prices[-1][0]}
    explanation = explain_decision(pred_price=forecast_prices.mean(), indicators=indicators)
    st.markdown("#### تفسير النموذج:")
    st.info(explanation)
