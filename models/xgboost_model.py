import xgboost as xgb
import numpy as np

def train_xgb(X_train, y_train, X_test):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train[:, -1])  # predict only the last point
    preds = model.predict(X_test)
    return np.tile(preds.reshape(-1, 1), (1, y_train.shape[1]))