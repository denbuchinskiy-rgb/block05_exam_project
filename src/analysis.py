import pandas as pd


def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Считает продажи по категориям."""
    result = (
        df.groupby("category", as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity", "sum"),
            orders=("product", "count"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    return result


def sales_by_manager(df: pd.DataFrame) -> pd.DataFrame:
    """Считает продажи по менеджерам."""
    result = (
        df.groupby("manager", as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity", "sum"),
            orders=("product", "count"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    return result


def sales_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Считает продажи по месяцам."""
    result = (
        df.groupby("month", as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity", "sum"),
            orders=("product", "count"),
        )
        .sort_values("month")
    )
    return result


def pivot_city_category(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт сводную таблицу: город × категория."""
    pivot = pd.pivot_table(
        df,
        index="city",
        columns="category",
        values="revenue",
        aggfunc="sum",
        fill_value=0,
    )
    return pivot
