"""Модуль очистки данных."""

from __future__ import annotations

import pandas as pd


class DataCleaner:
    """Проверка и очистка DataFrame."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()

    def validate_columns(self, required_columns: list[str]) -> None:
        """Проверить, что в таблице есть обязательные колонки."""
        missing_columns = set(required_columns) - set(self.dataframe.columns)
        if missing_columns:
            raise ValueError(f"В датасете нет колонок: {missing_columns}")
        print(required_columns)
    def fill_missing_quantity(self) -> None:
        """Заполнить пропуски в quantity медианным значением."""
        if "quantity" not in self.dataframe.columns:
            return

        median_value = self.dataframe["quantity"].median()
        self.dataframe["quantity"] = self.dataframe["quantity"].fillna(median_value)

    def drop_duplicates(self) -> None:
        """Удалить полностью повторяющиеся строки."""
        self.dataframe = self.dataframe.drop_duplicates().reset_index(drop=True)

    def clean(self, required_columns: list[str]) -> pd.DataFrame:
        """Полный сценарий очистки."""
        self.validate_columns(required_columns)
        self.fill_missing_quantity()
        self.drop_duplicates()

        total_missing = int(self.dataframe.isna().sum().sum())
        if total_missing != 0:
            raise ValueError(f"После очистки остались пропуски: {total_missing}")

        return self.dataframe.copy()