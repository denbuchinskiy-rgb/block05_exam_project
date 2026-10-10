"""Конфигурация проекта DataAnalyzer (блок 6)."""

from pathlib import Path

# Корневая папка проекта.
BASE_DIR = Path(__file__).resolve().parent.parent

# Папки проекта.
DATA_DIR = BASE_DIR / "data"
print(DATA_DIR)
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ML_DATA_DIR = DATA_DIR / "ml"
REPORTS_DIR = BASE_DIR / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

# Локальный путь к исходному датасету.
RAW_DATA_PATH = DATA_DIR / "sales_data.csv"
print(RAW_DATA_PATH)
# Основные выходные файлы.
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "sales_clean.csv"
GROUP_REPORT_PATH = REPORTS_DIR / "group_report_category.csv"
CORRELATION_PATH = REPORTS_DIR / "correlation.csv"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"

# Файлы для машинного обучения.
X_TRAIN_PATH = ML_DATA_DIR / "X_train.csv"
X_TEST_PATH = ML_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = ML_DATA_DIR / "y_train.csv"
Y_TEST_PATH = ML_DATA_DIR / "y_test.csv"

# Обязательные колонки нового датасета продаж.
REQUIRED_COLUMNS = [
    "number",
    "manager",
    "city",
    "category",
    "product",
    "quantity",
    "price",
    "discount",
]
print(REQUIRED_COLUMNS)
# Целевая переменная для регрессии.
TARGET_COLUMN = "price"

# Категориальный столбец.
CATEGORICAL_COLUMN = "category"

# Параметры повторяемости.
RANDOM_STATE = 42
TEST_SIZE = 0.20