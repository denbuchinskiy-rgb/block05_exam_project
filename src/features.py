import pandas as pd
 
 
def prepare_sales_features(df, target_column="quantity"):
    """Подготовить признаки X и целевую переменную y.
 
    Важная идея:
    - дату нельзя напрямую передавать в StandardScaler;
    - текстовые колонки нельзя напрямую передавать в LinearRegression;
    - поэтому дату превращаем в числовые признаки,
      а категории позже кодируем через OneHotEncoder.
    """
    data = df.copy()
 
    data["date"] = pd.to_datetime(data["number"])
 
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["dayofweek"] = data["date"].dt.dayofweek
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
 
    # Удаляем строки без целевой переменной и важных признаков.
    data = data.dropna(
        subset=[
            "manager",
            "city",
            "category",
            "product",
            "quantity",
            "price",
            "discount",
            "month",
            "day",
            "dayofweek",
            "is_weekend",
        ]
    ).copy()
 
    X = data[
        [
            "manager",
            "city",
            "category",
            "product",
            "price",
            "discount",
            "month",
            "day",
            "dayofweek",
            "is_weekend",
        ]
    ]
 
    y = data[target_column]
 
    return X, y, data
