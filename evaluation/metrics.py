
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    direction_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
    return {"RMSE": rmse, "MAE": mae, "Direction_Accuracy": direction_acc}
