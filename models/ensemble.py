
import numpy as np

def ensemble_predict(models, X):
    preds = [model.predict(X) for model in models]
    return np.mean(preds, axis=0)
