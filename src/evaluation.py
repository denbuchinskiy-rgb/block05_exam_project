import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
 
def calculate_regression_metrics(y_true, y_pred):
    """Посчитать метрики регрессии: MAE, MSE, RMSE, R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
 
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }
 
 
def print_metrics(metrics):
    """Вывести метрики в понятном виде."""
    print("Метрики качества:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
