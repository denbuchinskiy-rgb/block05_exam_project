"""Модуль загрузки данных.

Отвечает только за загрузку и сохранение CSV.
"""

import pandas as pd
 
 
def load_sales_data(file_path, sheet_name="sales_data"):
    """Загрузить Excel-файл с продажами.
 
    Ожидается лист sales_data и колонки:
    date, manager, city, category, product, quantity, price, discount.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df
 
 
def validate_columns(df, required_columns):
    """Проверить наличие обязательных колонок."""
    missing_columns = set(required_columns) - set(df.columns)
 
    if missing_columns:
        raise ValueError(f"В датасете нет колонок: {missing_columns}")
 
    return True
