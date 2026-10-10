from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
 
def split_data(X, y, test_size=0.20, random_state=42):
    """Разделить данные на train/test."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
 
 
def build_preprocessor(numeric_features, categorical_features):
    """Собрать preprocessing для числовых и категориальных признаков.
 
    Числовые признаки:
    - заполнение пропусков медианой;
    - масштабирование StandardScaler.
 
    Категориальные признаки:
    - заполнение самым частым значением;
    - OneHotEncoder.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
 
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
 
    return preprocessor
 
 
def build_regression_pipeline(
    numeric_features,
    categorical_features,
    model_type="linear",
    random_state=42,
):
    """Создать Pipeline: preprocessing + модель регрессии."""
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
 
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=1.0, random_state=random_state)
    elif model_type == "forest":
        model = RandomForestRegressor(
            n_estimators=150,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Неизвестный model_type: {model_type}")
 
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
 
    return pipeline
