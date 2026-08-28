import pandas as pd


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает таблицу продаж.

    Что делает функция:
    1. Копирует DataFrame.
    2. Приводит дату к типу datetime.
    3. Удаляет строки без ключевых данных.
    4. Заполняет скидку нулём.
    5. Удаляет дубликаты.
    6. Создаёт вычисляемые столбцы.
    """
    df = df.copy()

    required_columns = [
        "date", "manager", "city", "category",
        "product", "quantity", "price", "discount",
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Не хватает столбцов: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "manager", "category", "product"])
    df["discount"] = df["discount"].fillna(0)
    df["quantity"] = df["quantity"].fillna(0)
    df["price"] = df["price"].fillna(0)
    df = df.drop_duplicates()

    df["revenue_before_discount"] = df["quantity"] * df["price"]
    df["discount_amount"] = df["revenue_before_discount"] * df["discount"]
    df["revenue"] = df["revenue_before_discount"] - df["discount_amount"]
    df["month"] = df["date"].dt.to_period("M").astype(str)

    return df
