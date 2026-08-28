from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: str | Path,
) -> None:
    """Сохраняет столбчатую диаграмму."""
    output_path = Path(output_path)

    plt.figure(figsize=(10, 5))
    plt.bar(df[x_column], df[y_column])
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: str | Path,
) -> None:
    """Сохраняет линейный график."""
    output_path = Path(output_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df[x_column], df[y_column], marker="o")
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
