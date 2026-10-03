"""Project configuration for Block 6 DataAnalyzer."""
 
from pathlib import Path
 
# Корневая папка проекта.
BASE_DIR = Path(__file__).resolve().parent.parent
 
# Папки проекта.
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ML_DATA_DIR = DATA_DIR / "ml"
REPORTS_DIR = BASE_DIR / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

# Источник данных из финального ноутбука блока 6.
DATA_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml/"
    "master/datasets/housing/housing.csv"
)

# Локальный путь к исходному датасету.
RAW_DATA_PATH = DATA_DIR / "sales_data.csv"
 
# Основные выходные файлы.
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "sales_clean.csv"
GROUP_REPORT_PATH = REPORTS_DIR / "group_report_category.csv"
NUMERIC_STATS_PATH = REPORTS_DIR / "numeric_statistics.csv"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"
 
# Файлы для будущего машинного обучения.
X_TRAIN_PATH = ML_DATA_DIR / "X_train.csv"
X_TEST_PATH = ML_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = ML_DATA_DIR / "y_train.csv"
Y_TEST_PATH = ML_DATA_DIR / "y_test.csv"
 
# Колонки датасета California Housing.
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
 
# Целевая переменная для будущей задачи регрессии.
TARGET_COLUMN = "price"
 
# Категориальный столбец.
CATEGORICAL_COLUMN = "category"
 
# Параметры повторяемости.
RANDOM_STATE = 42
TEST_SIZE = 0.20
