"""Data cleaning module."""
 
from __future__ import annotations
 
import pandas as pd
 
 
class DataCleaner:
    """Класс для проверки и очистки данных.
 
    Объект DataCleaner получает DataFrame и выполняет последовательную очистку:
    проверка колонок, заполнение пропусков, удаление дубликатов.
    """
 
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()
 
    def validate_columns(self, required_columns: list[str]) -> None:
        """Проверить, что в таблице есть обязательные колонки."""
        missing_columns = set(required_columns) - set(self.dataframe.columns)
 
        if missing_columns:
            raise ValueError(f"В датасете нет колонок: {missing_columns}")
 
    def fill_missing_total_bedrooms(self) -> None:
        """Заполнить пропуски в total_bedrooms медианным значением.
 
        В исходном датасете California Housing именно total_bedrooms
        обычно содержит пропущенные значения.
        """
        median_value = self.dataframe["category"].median()
        self.dataframe["category"] = (
            self.dataframe["category"].fillna(median_value)
        )
 
    def drop_duplicates(self) -> None:
        """Удалить полностью повторяющиеся строки."""
        self.dataframe = self.dataframe.drop_duplicates()
 
    def clean(self, required_columns: list[str]) -> pd.DataFrame:
        """Выполнить полный сценарий очистки."""
        self.validate_columns(required_columns)
        self.fill_missing_total_bedrooms()
        self.drop_duplicates()
 
        total_missing = int(self.dataframe.isna().sum().sum())
        if total_missing != 0:
            raise ValueError(f"После очистки остались пропуски: {total_missing}")
 
        return self.dataframe.copy()
