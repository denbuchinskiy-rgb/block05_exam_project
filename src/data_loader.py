"""Модуль загрузки данных.

Отвечает только за загрузку и сохранение CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class DataLoader:
    """Класс для загрузки CSV-датасета продаж."""

    def __init__(self, raw_path: Path):
        self.raw_path = Path(raw_path)

    def load(self) -> pd.DataFrame:
        """Загрузить датасет.

        1. Если файл есть — читаем.
        2. Если нет — генерируем демо-датасет и сохраняем.
        """
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        print(self.raw_path)
        if self.raw_path.exists():
            return pd.read_csv(
                self.raw_path,
                sep=';',
                decimal=',',
                thousands='.',
                engine='python',
            )

        data = self.create_demo_sales_dataset()
        data.to_csv(self.raw_path, index=False, sep=';')
        return data

    def save_dataframe(self, dataframe: pd.DataFrame, path: Path) -> None:
        """Сохранить DataFrame в CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)

    @staticmethod
    def create_demo_sales_dataset(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
        """Учебный датасет продаж.

        Колонки: number, manager, city, category, product,
        quantity, price, discount.
        """
        rng = np.random.default_rng(seed)

        category_values = ["Одежда", "Книги", "Дом и сад", "Продукты"]
        category = rng.choice(category_values, size=rows, p=[0.42, 0.34, 0.12, 0.12])

        manager_values = ["Иванов", "Петров", "Сидоров", "Кузнецова"]
        manager = rng.choice(manager_values, size=rows)

        city_values = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
        city = rng.choice(city_values, size=rows)

        product_values = ["Футболка", "Роман", "Лампа", "Молоко", "Куртка", "Учебник"]
        product = rng.choice(product_values, size=rows)

        quantity = rng.gamma(shape=4.0, scale=1.2, size=rows)
        quantity = np.clip(quantity, 0.5, 15.0)

        discount = rng.integers(0, 30, size=rows)          # процент скидки
        price = rng.integers(100, 12000, size=rows)        # цена единицы товара

        # Немного пропусков в quantity, чтобы было что очищать.
        missing_mask = rng.random(rows) < 0.04
        quantity = quantity.astype(float)
        quantity[missing_mask] = np.nan

        number = np.arange(1, rows + 1)

        return pd.DataFrame(
            {
                "number": number,
                "manager": manager,
                "city": city,
                "category": category,
                "product": product,
                "quantity": quantity,
                "price": price,
                "discount": discount,
            }
        )