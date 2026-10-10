from pathlib import Path
 
 
def save_markdown_report(
    output_path,
    metrics,
    model_name,
    target_column,
    numeric_features,
    categorical_features,
):
    """Сохранить итоговый markdown-отчёт проекта."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    report_text = f'''# Отчёт по экзаменационному проекту блока 7
 
## Тема
Прогнозирование количества проданных единиц товара.
 
## Тип машинного обучения
Регрессия.
 
## Целевая переменная
`{target_column}`
 
## Модель
`{model_name}`
 
## Числовые признаки
{numeric_features}
 
## Категориальные признаки
{categorical_features}
 
## Метрики качества
- MAE: {metrics["MAE"]:.4f}
- MSE: {metrics["MSE"]:.4f}
- RMSE: {metrics["RMSE"]:.4f}
- R2: {metrics["R2"]:.4f}
 
## Интерпретация
MAE показывает среднюю абсолютную ошибку прогноза.
RMSE сильнее реагирует на крупные ошибки.
R2 показывает, насколько модель лучше простого прогноза среднего значения.
 
## Важное замечание
Дата была преобразована в числовые признаки: month, day, dayofweek, is_weekend.
Текстовые признаки были закодированы через OneHotEncoder.
StandardScaler применялся только к числовым признакам.
 
## Связь с дипломом
Этот проект можно использовать как основу для диплома по прогнозированию:
продажи, финансы, питание, вовлечённость в мероприятия или производственные показатели.
'''
 
    output_path.write_text(report_text, encoding="utf-8")
