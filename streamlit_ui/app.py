
import streamlit as st
from core.auto_predictor import predict_symbol

st.set_page_config(page_title="OmniMarket Prophet", layout="centered")
st.title("🔮 OmniMarket Prophet")

symbol = st.text_input("أدخل رمز السهم (مثل AAPL، TSLA):", value="AAPL")

if st.button("تنفيذ التوقع"):
    try:
        with st.spinner("جاري التنبؤ..."):
            result = predict_symbol(symbol)
            st.success(f"تم التنبؤ لـ {result['symbol']}")
            st.metric("السعر المتوقع", f"${result['prediction']:.2f}")
            st.metric("السعر الفعلي", f"${result['actual']:.2f}")
            st.metric("الفرق", f"${result['difference']:.2f}")
    except Exception as e:
        st.error(f"خطأ أثناء التنبؤ: {e}")
