from pathlib import Path

from src.load_data import load_excel_data
from src.clean_data import clean_sales_data
from src.analysis import (
    sales_by_category,
    sales_by_manager,
    sales_by_month,
    pivot_city_category,
)
from src.charts import save_bar_chart, save_line_chart


DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")


def main() -> None:
    """Главный сценарий проекта."""
    REPORTS_DIR.mkdir(exist_ok=True)

    input_path = DATA_DIR / "sales_data.xlsx"

    print("1. Загружаем данные...")
    df_raw = load_excel_data(input_path)

    print("2. Очищаем данные...")
    df_clean = clean_sales_data(df_raw)

    print("3. Считаем отчёты...")
    category_report = sales_by_category(df_clean)
    manager_report = sales_by_manager(df_clean)
    month_report = sales_by_month(df_clean)
    pivot_report = pivot_city_category(df_clean)

    print("4. Сохраняем CSV-отчёты...")
    df_clean.to_csv(REPORTS_DIR / "cleaned_sales_data.csv", index=False)
    category_report.to_csv(REPORTS_DIR / "sales_by_category.csv", index=False)
    manager_report.to_csv(REPORTS_DIR / "sales_by_manager.csv", index=False)
    month_report.to_csv(REPORTS_DIR / "sales_by_month.csv", index=False)
    pivot_report.to_csv(REPORTS_DIR / "pivot_city_category.csv")

    print("5. Строим графики...")
    save_bar_chart(
        category_report,
        x_column="category",
        y_column="total_revenue",
        title="Продажи по категориям",
        output_path=REPORTS_DIR / "chart_sales_by_category.png",
    )

    save_bar_chart(
        manager_report,
        x_column="manager",
        y_column="total_revenue",
        title="Продажи по менеджерам",
        output_path=REPORTS_DIR / "chart_sales_by_manager.png",
    )

    save_line_chart(
        month_report,
        x_column="month",
        y_column="total_revenue",
        title="Продажи по месяцам",
        output_path=REPORTS_DIR / "chart_sales_by_month.png",
    )

    print("6. Формируем итоговый markdown-отчёт...")

    total_revenue = df_clean["revenue"].sum()
    total_quantity = df_clean["quantity"].sum()
    best_category = category_report.iloc[0]["category"]
    best_manager = manager_report.iloc[0]["manager"]

    report_text = f"""# Итоговый отчёт по проекту блока 5

## 1. Данные

Источник данных: `data/sales_data.xlsx`.
Количество строк после очистки: {len(df_clean)}.

## 2. Основные показатели

Общая выручка: {total_revenue:,.2f}
Общее количество проданных товаров: {total_quantity:,.0f}
Лучшая категория по выручке: {best_category}
Лучший менеджер по выручке: {best_manager}

## 3. Что было сделано

1. Данные загружены из Excel.
2. Проверены обязательные столбцы.
3. Обработаны пропуски.
4. Удалены дубликаты.
5. Созданы вычисляемые столбцы.
6. Сделаны группировки.
7. Создана сводная таблица город × категория.
8. Построены графики.

## 4. Связь Excel и Python

Excel удобен для визуальной проверки таблицы.
Python удобен для повторяемого анализа и автоматизации.
"""

    report_path = REPORTS_DIR / "final_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    print("Готово.")
    print(f"Отчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
