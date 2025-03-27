
import ta

def generate_features(df):
    df["SMA_10"] = ta.trend.SMAIndicator(df["Close"], window=10).sma_indicator()
    df["RSI_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["Volatility"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df.fillna(method="bfill", inplace=True)
    return df
