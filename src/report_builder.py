"""Модуль формирования markdown-отчёта."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ReportBuilder:
    """Формирует итоговый markdown-отчёт."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

    def build_report(
        self,
        basic_info: dict[str, object],
        target_stats: dict[str, float],
        group_report: pd.DataFrame,
        correlation: pd.Series,
        ml_shapes: dict[str, tuple[int, int] | int],
    ) -> str:
        """Сформировать текст отчёта."""
        top_category = group_report["mean"].idxmax()

        report_text = f"""# Финальный отчёт проекта DataAnalyzer

## 1. Назначение проекта

Проект выполнен в рамках блока 6 «Погружение в Python и анализ данных».
Показан полный цикл: загрузка, очистка, анализ, визуализация,
отчёт и подготовка к машинному обучению.

## 2. Данные

Используется датасет продаж `sales_data.csv`.

- Количество строк после очистки: {basic_info["rows"]}
- Количество колонок после очистки: {basic_info["columns"]}

## 3. Статистика цены

- Минимум: {target_stats["min"]:.2f}
- Максимум: {target_stats["max"]:.2f}
- Среднее: {target_stats["mean"]:.2f}
- Медиана: {target_stats["median"]:.2f}
- Стандартное отклонение: {target_stats["std"]:.2f}

## 4. Групповой анализ

Группировка по колонке `category`, целевой показатель — `price`.
Самая высокая средняя цена в группе: **{top_category}**.

## 5. Корреляция

Корреляции числовых признаков с целевой переменной `price`:

{correlation.head(5).to_string()}

## 6. Подготовка к машинному обучению

Созданы новые признаки:

- `revenue` — выручка без скидки;
- `discounted_price` — цена со скидкой;
- `discounted_revenue` — выручка со скидкой.

Категориальный столбец `category` закодирован One-Hot.

Данные разделены на train/test:

- X_train: {ml_shapes["X_train"]}
- X_test:  {ml_shapes["X_test"]}
- y_train: {ml_shapes["y_train"]}
- y_test:  {ml_shapes["y_test"]}

## 7. Связь с ИИ

Модель пока не обучается, но датасет уже подготовлен
для регрессии: предсказание `price` по остальным признакам.

## 8. Вывод

Проект демонстрирует архитектуру реального Python-проекта:
загрузка, очистка, анализ, визуализация и ML-подготовка
разнесены по отдельным классам.
"""
        return report_text

    def save(self, report_text: str) -> None:
        """Сохранить markdown-отчёт."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(report_text, encoding="utf-8")