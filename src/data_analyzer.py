"""Модуль анализа данных."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataAnalyzer:
    """Аналитик данных: NumPy + Pandas."""

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe должен быть pandas.DataFrame")
        self.dataframe = dataframe.copy()

    def basic_info(self) -> dict[str, object]:
        """Базовая информация о таблице."""
        return {
            "rows": int(self.dataframe.shape[0]),
            "columns": int(self.dataframe.shape[1]),
            "column_names": list(self.dataframe.columns),
        }

    def _check_column(self, column: str) -> None:
        if column not in self.dataframe.columns:
            raise KeyError(f"Колонка не найдена: {column!r}")

    def numeric_statistics(self, column: str) -> dict[str, float]:
        """NumPy-статистика по числовой колонке."""
        self._check_column(column)

        if not pd.api.types.is_numeric_dtype(self.dataframe[column]):
            raise TypeError(f"Колонка {column!r} должна быть числовой")

        values = self.dataframe[column].dropna().to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"Колонка {column!r} не содержит числовых значений")

        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
        }

    def pandas_describe(self) -> pd.DataFrame:
        """describe() для числовых колонок."""
        return self.dataframe.select_dtypes(include="number").describe()

    def group_report(
        self,
        group_column: str,
        value_column: str,
    ) -> pd.DataFrame:
        """Групповой отчёт: категория → статистика по цене."""
        self._check_column(group_column)
        self._check_column(value_column)

        if not pd.api.types.is_numeric_dtype(self.dataframe[value_column]):
            raise TypeError(f"Колонка {value_column!r} должна быть числовой")

        report = (
            self.dataframe.groupby(group_column)[value_column]
            .agg(["count", "mean", "min", "max"])
            .sort_values(by="mean", ascending=False)
        )
        return report

    def correlation_with_target(self, target_column: str) -> pd.Series:
        """Корреляции числовых признаков с целевой переменной."""
        self._check_column(target_column)

        numeric_df = self.dataframe.select_dtypes(include="number")
        if target_column not in numeric_df.columns:
            raise TypeError(f"Целевая колонка {target_column!r} должна быть числовой")

        return numeric_df.corr()[target_column].sort_values(ascending=False)