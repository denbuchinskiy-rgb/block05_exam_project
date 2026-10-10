"""Конфигурация проекта DataAnalyzer (блок 7)."""

from pathlib import Path
 
BASE_DIR = Path(__file__).resolve().parent.parent
 
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "sales_data.xlsx"
SHEET_NAME = "sales_data"
 
REPORTS_DIR = BASE_DIR / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"
PROCESSED_DIR = DATA_DIR / "processed"
 
TARGET_COLUMN = "quantity"
 
RAW_COLUMNS = [
    "number",
    "manager",
    "city",
    "category",
    "product",
    "quantity",
    "price",
    "discount",
]
 
NUMERIC_FEATURES = [
    "price",
    "discount",
    "month",
    "day",
    "dayofweek",
    "is_weekend",
]
 
CATEGORICAL_FEATURES = [
    "manager",
    "city",
    "category",
    "product",
]
 
TEST_SIZE = 0.20
RANDOM_STATE = 42
