# Приоритизация обращений

Ранжирование лидов по вероятности целевого действия в течение 5 дней после назначения.
Метрика — Daily Average Precision (AP считается внутри каждого дня, затем усредняется).

## Установка

```bash
pip install -r requirements.txt
```

## Данные

Положить рядом с ноутбуками:

```
data/
  train.csv     обучающая выборка с target
  test.csv      тестовая выборка без target
  events.csv    лог событий для feature engineering
```

## Запуск

```bash
jupyter lab
```

| файл | что делает |
|---|---|
| `eda.ipynb` | разведочный анализ, все выводы внутри. Для получения сабмита запускать не нужно |
| `main.ipynb` | валидация, сравнение моделей, обучение финальной модели, запись `submission.csv` |
| `preprocessor.py` | feature engineering, импортируется из `main.ipynb` |

Для сабмита: выполнить `main.ipynb` сверху вниз. На выходе — `submission.csv` с колонками `lead_id` и `score`.

**Время прогона**: с включённым подбором гиперпараметров (`tune` в `walk_forward`) — десятки минут. Чтобы быстро проверить работоспособность, поставьте `N_TRIALS = 3`.

**Использованные open-source библиотеки**: pandas, numpy, scipy, scikit-learn, CatBoost, Optuna, matplotlib, seaborn.
