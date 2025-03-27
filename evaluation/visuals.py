
import matplotlib.pyplot as plt

def plot_prediction(y_true, y_pred):
    plt.figure(figsize=(12,6))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Prediction vs Actual")
    plt.show()
