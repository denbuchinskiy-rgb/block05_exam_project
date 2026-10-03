"""Data loading module.
 
Этот модуль отвечает только за загрузку и сохранение данных.
Такой подход делает проект понятным: загрузка не смешивается с очисткой,
анализом, графиками и подготовкой к машинному обучению.
"""
 
from __future__ import annotations
 
from pathlib import Path
 
import numpy as np
import pandas as pd
 
 
class DataLoader:
    """Класс для загрузки CSV-датасета.
 
    Класс хранит путь к файлу и ссылку на внешний источник.
    Метод load() возвращает pandas DataFrame.
    """
 
    def __init__(self, raw_path: Path, url: str):
        self.raw_path = Path(raw_path)
        self.url = url
 
    def load(self) -> pd.DataFrame:
        """Загрузить датасет.
 
        Алгоритм:
        1. Если файл уже есть в data/housing.csv, читаем его.
        2. Если файла нет, пробуем скачать CSV по URL.
        3. Если интернет недоступен, создаём учебный fallback-датасет.
        """
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
 
        if self.raw_path.exists():
            return pd.read_csv(self.raw_path)
 
        try:
            data = pd.read_csv(self.url)
            data.to_csv(self.raw_path, index=False)
            return data
        except Exception:
            data = self.create_demo_housing_dataset()
            data.to_csv(self.raw_path, index=False)
            return data
 
    def save_dataframe(self, dataframe: pd.DataFrame, path: Path) -> None:
        """Сохранить DataFrame в CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)
 
    @staticmethod
    def create_demo_housing_dataset(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
        """Создать учебный датасет, похожий по колонкам на California Housing.
 
        Это запасной вариант для учебной аудитории, если интернет не работает.
        Для экзамена желательно использовать реальный CSV из URL.
        """
        rng = np.random.default_rng(seed)
 
        category_values = ["Одежда", "Книги", "Дом и сад", "Продукты"]
        category = rng.choice(
            category_values,
            size=rows,
            p=[0.42, 0.34, 0.12, 0.12],
        )
 
        quantity = rng.gamma(shape=4.0, scale=1.2, size=rows)
        quantity = np.clip(quantity, 0.5, 15.0)
 
        discount = rng.integers(80, 2500, size=rows)
        price = discount * rng.normal(2.8, 0.6, size=rows)
        price = np.clip(price, 100, 12000).astype(int)
 
        # Добавим немного пропусков, чтобы было что очищать.
        missing_mask = rng.random(rows) < 0.04
        product = product.astype(float)
        product[missing_mask] = np.nan
 
        bonus = np.where(category == "Электроника", -45000, 35000)
        house_value = (
            50000
            + quantity * 42000
            + bonus
            + rng.normal(0, 35000, size=rows)
        )
        house_value = np.clip(house_value, 35000, 500001)
 
        return pd.DataFrame(
            {
                "category": category, 
                "product": product, 
                "quantity": pd.array(quantity, dtype="Int64"), # Int64 поддерживает пропуски (pd.NA) "price": price, "discount": discount,
            }
        )
