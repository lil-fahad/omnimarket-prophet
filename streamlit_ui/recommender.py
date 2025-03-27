
import streamlit as st
import pandas as pd
from core.auto_predictor import predict_symbol

st.header("🧠 توصيات التداول الذكية")

symbols = st.text_area("أدخل رموز الأسهم (كل رمز في سطر):", "AAPL
TSLA
MSFT").splitlines()
recommendations = []

if st.button("تحليل وتوصية"):
    st.write("جارٍ التقييم ...")
    for symbol in symbols:
        try:
            result = predict_symbol(symbol)
            confidence = abs(result["difference"]) / (result["actual"] + 1e-5)
            recommendations.append({
                "رمز السهم": result["symbol"],
                "السعر المتوقع": round(result["prediction"], 2),
                "السعر الحالي": round(result["actual"], 2),
                "فرق السعر": round(result["difference"], 2),
                "نسبة الثقة": round(confidence * 100, 2),
                "توصية": "شراء" if result["difference"] > 0 else "بيع"
            })
        except Exception as e:
            st.error(f"⚠️ {symbol}: {e}")

    if recommendations:
        df = pd.DataFrame(recommendations)
        df = df.sort_values(by="نسبة الثقة", ascending=False)
        st.subheader("📊 توصيات مرتبة حسب نسبة الثقة")
        st.dataframe(df)
