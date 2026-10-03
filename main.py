# from pathlib import Path

# from src.load_data import load_excel_data
# from src.clean_data import clean_sales_data
# from src.analysis import (
#     sales_by_category,
#     sales_by_manager,
#     sales_by_month,
#     pivot_city_category,
# )
# from src.charts import save_bar_chart, save_line_chart


# DATA_DIR = Path("data")
# REPORTS_DIR = Path("reports")


# def main() -> None:
#     """Главный сценарий проекта."""
#     REPORTS_DIR.mkdir(exist_ok=True)

#     input_path = DATA_DIR / "sales_data.xlsx"

#     print("1. Загружаем данные...")
#     df_raw = load_excel_data(input_path)

#     print("2. Очищаем данные...")
#     df_clean = clean_sales_data(df_raw)

#     print("3. Считаем отчёты...")
#     category_report = sales_by_category(df_clean)
#     manager_report = sales_by_manager(df_clean)
#     month_report = sales_by_month(df_clean)
#     pivot_report = pivot_city_category(df_clean)

#     print("4. Сохраняем CSV-отчёты...")
#     df_clean.to_csv(REPORTS_DIR / "cleaned_sales_data.csv", index=False)
#     category_report.to_csv(REPORTS_DIR / "sales_by_category.csv", index=False)
#     manager_report.to_csv(REPORTS_DIR / "sales_by_manager.csv", index=False)
#     month_report.to_csv(REPORTS_DIR / "sales_by_month.csv", index=False)
#     pivot_report.to_csv(REPORTS_DIR / "pivot_city_category.csv")

#     print("5. Строим графики...")
#     save_bar_chart(
#         category_report,
#         x_column="category",
#         y_column="total_revenue",
#         title="Продажи по категориям",
#         output_path=REPORTS_DIR / "chart_sales_by_category.png",
#     )

#     save_bar_chart(
#         manager_report,
#         x_column="manager",
#         y_column="total_revenue",
#         title="Продажи по менеджерам",
#         output_path=REPORTS_DIR / "chart_sales_by_manager.png",
#     )

#     save_line_chart(
#         month_report,
#         x_column="month",
#         y_column="total_revenue",
#         title="Продажи по месяцам",
#         output_path=REPORTS_DIR / "chart_sales_by_month.png",
#     )

#     print("6. Формируем итоговый markdown-отчёт...")

#     total_revenue = df_clean["revenue"].sum()
#     total_quantity = df_clean["quantity"].sum()
#     best_category = category_report.iloc[0]["category"]
#     best_manager = manager_report.iloc[0]["manager"]

#     report_text = f"""# Итоговый отчёт по проекту блока 5

# ## 1. Данные

# Источник данных: `data/sales_data.xlsx`.
# Количество строк после очистки: {len(df_clean)}.

# ## 2. Основные показатели

# Общая выручка: {total_revenue:,.2f}
# Общее количество проданных товаров: {total_quantity:,.0f}
# Лучшая категория по выручке: {best_category}
# Лучший менеджер по выручке: {best_manager}

# ## 3. Что было сделано

# 1. Данные загружены из Excel.
# 2. Проверены обязательные столбцы.
# 3. Обработаны пропуски.
# 4. Удалены дубликаты.
# 5. Созданы вычисляемые столбцы.
# 6. Сделаны группировки.
# 7. Создана сводная таблица город × категория.
# 8. Построены графики.

# ## 4. Связь Excel и Python

# Excel удобен для визуальной проверки таблицы.
# Python удобен для повторяемого анализа и автоматизации.
# """

#     report_path = REPORTS_DIR / "final_report.md"
#     report_path.write_text(report_text, encoding="utf-8")

#     print("Готово.")
#     print(f"Отчёт сохранён: {report_path}")


# if __name__ == "__main__":
#     main()

"""Main script for Block 6 final DataAnalyzer project."""
 
from src.config import (
    CATEGORICAL_COLUMN,
    CHARTS_DIR,
    CLEAN_DATA_PATH,
    DATA_URL,
    FINAL_REPORT_PATH,
    GROUP_REPORT_PATH,
    ML_DATA_DIR,
    NUMERIC_STATS_PATH,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.data_analyzer import DataAnalyzer
from src.ml_preparer import MLDatasetPreparer
from src.visualizer import Visualizer
from src.report_builder import ReportBuilder
 
 
def main() -> None:
    """Run the full DataAnalyzer project."""
 
    # 1. Create folders.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)
 
    # 2. Load data.
    print("1. Loading dataset...")
    loader = DataLoader(raw_path=RAW_DATA_PATH, url=DATA_URL)
    df_raw = loader.load()
    print("Raw shape:", df_raw.shape)
 
    # 3. Clean data.
    print("2. Cleaning dataset...")
    cleaner = DataCleaner(df_raw)
    df_clean = cleaner.clean(required_columns=REQUIRED_COLUMNS)
    loader.save_dataframe(df_clean, CLEAN_DATA_PATH)
    print("Clean shape:", df_clean.shape)
 
    # 4. Analyze data.
    print("3. Analyzing dataset...")
    analyzer = DataAnalyzer(df_clean)
 
    basic_info = analyzer.basic_info()
    house_value_stats = analyzer.numeric_statistics(TARGET_COLUMN)
    group_report = analyzer.group_report(CATEGORICAL_COLUMN, TARGET_COLUMN)
    correlation = analyzer.correlation_with_target(TARGET_COLUMN)
 
    group_report.to_csv(GROUP_REPORT_PATH)
    correlation.to_csv(NUMERIC_STATS_PATH)
 
    print("House value stats:", house_value_stats)
    print("Group report:")
    print(group_report)
 
    # 5. Save charts.
    print("4. Saving charts...")
    visualizer = Visualizer(CHARTS_DIR)
 
    visualizer.save_histogram(
        dataframe=df_clean,
        column=TARGET_COLUMN,
        filename="house_value_histogram.png",
        title="Распределение стоимости жилья",
    )
 
    visualizer.save_scatter(
        dataframe=df_clean,
        x_column="median_income",
        y_column=TARGET_COLUMN,
        filename="income_vs_house_value.png",
        title="Доход и стоимость жилья",
    )
 
    visualizer.save_group_bar(
        group_report=group_report,
        value_column="mean",
        filename="mean_house_value_by_ocean.png",
        title="Средняя стоимость жилья по близости к океану",
    )
 
    # 6. Prepare dataset for machine learning.
    print("5. Preparing ML dataset...")
    ml_preparer = MLDatasetPreparer(
        dataframe=df_clean,
        target_column=TARGET_COLUMN,
        categorical_column=CATEGORICAL_COLUMN,
        random_state=RANDOM_STATE,
    )
 
    X_train, X_test, y_train, y_test = ml_preparer.prepare(
        test_size=TEST_SIZE,
        scale=True,
    )
 
    ml_preparer.save_prepared_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=ML_DATA_DIR,
    )
 
    ml_shapes = {
        "X_train": X_train.shape,
        "X_test": X_test.shape,
        "y_train": len(y_train),
        "y_test": len(y_test),
    }
 
    print("ML shapes:", ml_shapes)
 
    # 7. Build report.
    print("6. Building final report...")
    report_builder = ReportBuilder(FINAL_REPORT_PATH)
 
    report_text = report_builder.build_report(
        basic_info=basic_info,
        house_value_stats=house_value_stats,
        group_report=group_report,
        correlation=correlation,
        ml_shapes=ml_shapes,
    )
    report_builder.save(report_text)
 
    print("Project completed.")
    print("Final report:", FINAL_REPORT_PATH)
 
 
if __name__ == "__main__":
    main()
