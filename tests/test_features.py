import pandas as pd
 
from src.features import prepare_sales_features
 
 
def test_prepare_sales_features_creates_date_features():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-06"],
            "manager": ["Иванов А.С.", "Сидоров К.Д."],
            "city": ["Москва", "Казань"],
            "category": ["Продукты", "Книги"],
            "product": ["Шоколад", "Детектив"],
            "quantity": [3, 1],
            "price": [149.17, 734.32],
            "discount": [0.46, 0.40],
        }
    )
 
    X, y, data = prepare_sales_features(df)
 
    assert "month" in X.columns
    assert "day" in X.columns
    assert "dayofweek" in X.columns
    assert "is_weekend" in X.columns
    assert len(X) == len(y)
    assert len(data) == 2
