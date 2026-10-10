"""Visualization module."""
 
from pathlib import Path
 
import matplotlib.pyplot as plt
import pandas as pd
 
 
def save_fact_prediction_plot(y_true, y_pred, output_path):
    """Сохранить график факт vs прогноз."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.35)
    plt.xlabel("Факт")
    plt.ylabel("Прогноз")
    plt.title("Регрессия: факт vs прогноз")
    plt.grid(True)
 
    min_value = min(min(y_true), min(y_pred))
    max_value = max(max(y_true), max(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value])
 
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
 
 
def save_error_histogram(errors, output_path):
    """Сохранить гистограмму ошибок."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=30)
    plt.xlabel("Ошибка")
    plt.ylabel("Количество")
    plt.title("Распределение ошибок прогноза")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
 
 
def build_predictions_dataframe(y_true, y_pred):
    """Создать таблицу с фактом, прогнозом и ошибкой."""
    result = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
 
    result["error"] = result["y_true"] - result["y_pred"]
    result["absolute_error"] = result["error"].abs()
 
    return result
