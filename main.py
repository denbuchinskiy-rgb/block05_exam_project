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

"""Главный сценарий проекта DataAnalyzer (блок 6)."""

from IPython.display import display

from pathlib import Path
 
from src.config import (
    CATEGORICAL_FEATURES,
    CHARTS_DIR,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_COLUMNS,
    RAW_DATA_PATH,
    REPORTS_DIR,
    SHEET_NAME,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_sales_data, validate_columns
from src.evaluation import calculate_regression_metrics, print_metrics
from src.features import prepare_sales_features
from src.modeling import build_regression_pipeline, split_data
from src.reporting import save_markdown_report
from src.visualization import (
    build_predictions_dataframe,
    save_error_histogram,
    save_fact_prediction_plot,
)
 
 
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
    print("1. Загрузка данных")
    df = load_sales_data(RAW_DATA_PATH, sheet_name=SHEET_NAME)
    validate_columns(df, RAW_COLUMNS)
    print("Размер исходных данных:", df.shape)
 
    print("2. Подготовка признаков")
    X, y, prepared_data = prepare_sales_features(
        df,
        target_column=TARGET_COLUMN,
    )
 
    prepared_data.to_csv(
        PROCESSED_DIR / "prepared_sales_data.csv",
        index=False,
    )
 
    print("X:", X.shape)
    print("y:", y.shape)
 
    print("3. Train/test split")
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
 
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
 
    print("4. Обучение модели")
    model = build_regression_pipeline(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        model_type="linear",
        random_state=RANDOM_STATE,
    )
 
    model.fit(X_train, y_train)
 
    print("5. Прогноз")
    predictions = model.predict(X_test)
 
    print("Первые 5 прогнозов:")
    print(predictions[:5])
 
    print("6. Метрики")
    metrics = calculate_regression_metrics(y_test, predictions)
    print_metrics(metrics)
 
    print("7. Отчёты и графики")
    predictions_df = build_predictions_dataframe(
        y_true=y_test.reset_index(drop=True),
        y_pred=predictions,
    )
 
    predictions_df.to_csv(
        REPORTS_DIR / "predictions.csv",
        index=False,
    )
 
    save_fact_prediction_plot(
        y_true=predictions_df["y_true"],
        y_pred=predictions_df["y_pred"],
        output_path=CHARTS_DIR / "fact_vs_prediction.png",
    )
 
    save_error_histogram(
        errors=predictions_df["error"],
        output_path=CHARTS_DIR / "error_histogram.png",
    )
 
    save_markdown_report(
        output_path=REPORTS_DIR / "final_report.md",
        metrics=metrics,
        model_name="LinearRegression with preprocessing Pipeline",
        target_column=TARGET_COLUMN,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
 
    print("Проект выполнен.")
    print("Итоговый отчёт:", REPORTS_DIR / "final_report.md")
 
 
if __name__ == "__main__":
    main()
