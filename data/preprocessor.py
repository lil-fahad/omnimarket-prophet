
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prepare_sequences(df, feature_cols, target_col, window_size=60):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)

    X, y = [], []
    for i in range(window_size, len(df_scaled)):
        X.append(df_scaled.iloc[i - window_size:i].values)
        y.append(df[target_col].iloc[i])
    
    return np.array(X), np.array(y), scaler
