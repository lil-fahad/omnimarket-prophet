
import numpy as np
from models.lstm_advanced import build_lstm_advanced
from models.model_selector import select_model
from data.preprocessor import prepare_sequences
from data.features import generate_features
from data.polygon_api import fetch_polygon_data

def predict_symbol(symbol="AAPL"):
    # تحميل البيانات
    df = fetch_polygon_data(symbol=symbol)
    df = generate_features(df)

    # إعداد البيانات
    feature_cols = ["Close", "SMA_10", "RSI_14", "Volatility"]
    df = df.dropna()
    X, y, scaler = prepare_sequences(df, feature_cols, "Close", window_size=60)

    # اختيار النموذج الأفضل (أو استخدم LSTM متطور)
    model = build_lstm_advanced(input_shape=X.shape[1:])
    # هنا يجب تحميل الوزن المدرب مسبقاً أو تدريب سريع
    # model.load_weights('models/checkpoints/best_lstm.h5')

    # توقع
    preds = model.predict(X)
    latest_prediction = preds[-1][0]
    latest_actual = y[-1]
    
    result = {
        "symbol": symbol,
        "prediction": latest_prediction,
        "actual": latest_actual,
        "difference": latest_prediction - latest_actual
    }
    return result
