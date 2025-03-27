
import streamlit as st
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.auto_predictor import predict_symbol
from data.polygon_api import fetch_polygon_data
from data.features import generate_features
from data.preprocessor import prepare_sequences
from models.lstm_advanced import build_lstm_advanced

st.header("تفسير التوقع")

symbol = st.text_input("رمز السهم للتفسير", value="AAPL")
if st.button("حلل التفسير"):
    df = fetch_polygon_data(symbol)
    df = generate_features(df)
    df.dropna(inplace=True)
    feature_cols = ["Close", "SMA_10", "RSI_14", "Volatility"]
    X, y, scaler = prepare_sequences(df, feature_cols, "Close", window_size=60)

    model = build_lstm_advanced(input_shape=X.shape[1:])
    # تدريب نموذجي سريع لغرض العرض
    model.fit(X, y, epochs=1, batch_size=32, verbose=0)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X[:100])

    st.subheader("أهم العوامل التي أثرت في التوقع")
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(plt.gcf())
    plt.clf()
