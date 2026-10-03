"""Report builder module."""
 
from __future__ import annotations
 
from pathlib import Path
 
import pandas as pd
 
 
class ReportBuilder:
    """Класс для формирования итогового markdown-отчёта."""
 
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
 
    def build_report(
        self,
        basic_info: dict[str, object],
        house_value_stats: dict[str, float],
        group_report: pd.DataFrame,
        correlation: pd.Series,
        ml_shapes: dict[str, tuple[int, int] | int],
    ) -> str:
        """Сформировать текст отчёта."""
        top_category = group_report["mean"].idxmax()
 
        report_text = f"""# Финальный отчёт проекта DataAnalyzer
 
## 1. Назначение проекта
 
Проект выполнен в рамках блока 6 «Погружение в Python и анализ данных».
Он показывает полный цикл работы с табличным датасетом:
загрузка, очистка, анализ, визуализация, отчёт и подготовка к машинному обучению.
 
## 2. Данные
 
Используется датасет California Housing.
 
Количество строк после очистки: {basic_info["rows"]}.
Количество колонок после очистки: {basic_info["columns"]}.
 
## 3. Статистика стоимости жилья
 
Минимум: {house_value_stats["min"]:.2f}.
Максимум: {house_value_stats["max"]:.2f}.
Среднее: {house_value_stats["mean"]:.2f}.
Медиана: {house_value_stats["median"]:.2f}.
Стандартное отклонение: {house_value_stats["std"]:.2f}.
 
## 4. Групповой анализ
 
Самая высокая средняя стоимость жилья в группе: {top_category}.
 
Группировка выполнена по колонке `ocean_proximity`.
Целевой показатель: `median_house_value`.
 
## 5. Корреляция
 
Корреляция помогает понять, какие числовые признаки связаны
с целевой переменной `median_house_value`.
 
Самые важные корреляции:
 
{correlation.head(5).to_string()}
 
## 6. Подготовка к машинному обучению
 
Созданы новые признаки:
 
- rooms_per_household;
- bedrooms_per_room;
- population_per_household.
 
Категориальный столбец `ocean_proximity` преобразован в числовые 0/1-колонки.
 
Данные разделены на train/test:
 
- X_train: {ml_shapes["X_train"]}
- X_test: {ml_shapes["X_test"]}
- y_train: {ml_shapes["y_train"]}
- y_test: {ml_shapes["y_test"]}
 
## 7. Как это связано с ИИ
 
На этом этапе модель ещё не обучается.
Но данные уже подготовлены так, чтобы в следующем блоке использовать их
для задачи регрессии: прогнозировать `median_house_value`.
 
Для кластеризации можно использовать матрицу признаков X без целевой переменной y.
 
## 8. Вывод
 
Проект DataAnalyzer показывает архитектуру реального Python-проекта:
данные загружаются отдельным классом, очищаются отдельным классом,
анализируются отдельным классом, визуализируются отдельным классом
и готовятся к машинному обучению отдельным классом.
"""
        return report_text
 
    def save(self, report_text: str) -> None:
        """Сохранить markdown-отчёт."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(report_text, encoding="utf-8")
