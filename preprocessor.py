# файл содержит пайплайн предобработки данных (feature engineering)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "target"
ID_COLUMNS = ["lead_id", "user_id"]
TIME_COLUMNS = ["assignment_ts", "assignment_date"]
NON_FEATURE_COLUMNS = ID_COLUMNS + TIME_COLUMNS + [TARGET, "split"]

# признаки из events.csv, неплохо скоррелированные correlation ratio с target
EVENT_TYPES = ["item_view", "search", "favorite", "chat_open", "call_click"]

# окна для событийных счётчиков
# оставил только небольшие окна, для больших скор был хуже
# + глубже 29д истории в events.csv всё равно нет
EVENT_WINDOWS = [1, 3, 7]

# ЭВРИСТИКА (получилось - улучшила скор)
# действия более серьезного намерения - т.е. не просто просмотр или поиск - а именно связь/сохранение в избранные
# так вот такие действия должны по идее сильнее влиять на таргет - а их сумма тем более
HIGH_INTENT_TYPES = ["favorite", "chat_open", "call_click"]

# контексты показа (из events.csv)
CTX_VALUES = [f"c0{i}" for i in range(1, 9)]

# признаки, дублирующие другие (EDA1.2)
REDUNDANT_COLUMNS = ["is_weekend", "price_bucket"]

# категориальные признаки на выходе Preprocessor (для CatBoostClassifier)
CATEGORICAL_COLUMNS = ["lead_source", "call_center", "region",
                       "car_segment", "lead_channel", "user_tenure_bucket"]


class Preprocessor(BaseEstimator, TransformerMixin):
    """строит признаки из events.csv и чистит дубликаты

    Наследуемся от sklearn-базовых классов, чтобы объект можно было положить шагом
    в Pipeline: fit вызывается на train-части каждого фолда отдельно, и утечка
    между фолдами исключена по построению.

    transform принимает СЫРОЙ DataFrame строк (с lead_id и assignment_ts), а не срез
    по feature_columns: признакам из events нужны обе эти колонки.
    """

    def __init__(self, events=None, use_events=True, drop_redundant=True):
        self.events = events
        self.use_events = use_events
        self.drop_redundant = drop_redundant

    def fit(self, X, y=None):
        #  строит признаки, запоминает состав колонок и выбрасывает результат
        self._fitted_build(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        # переопределяем, потому что TransformerMixin.fit_transform это fit(X).transform(X),
        # т.е. два прогона _build с полным джойном events - вдвое дороже без причины
        return self._fitted_build(X)

    def transform(self, X):
        features = self._build(X)
        # порядок и состав колонок фиксируем по fit - на valid/test состав может
        # разойтись, если в срезе не встретился какой-то тип события
        return features.reindex(columns=self.feature_names_)

    def _fitted_build(self, X):
        features = self._build(X)
        self.feature_names_ = features.columns.tolist()
        return features

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)

    def _build(self, X):
        features = X.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in X.columns])

        if self.drop_redundant:
            features = features.drop(
                columns=[c for c in REDUNDANT_COLUMNS if c in features.columns]
            )
        if self.use_events and self.events is not None:
            features = features.join(self._event_features(X))

        return features

    def _event_features(self, X):
        """Агрегаты по событиям лида, случившимся СТРОГО ДО его назначения.
        Фильтр event_ts < assignment_ts обязателен для избежания утечки в будущее.
        """
        events = self.events[["lead_id", "event_ts", "event_type", "src_slot", "ctx_seq"]]
        joined = events.merge(X[["lead_id", "assignment_ts"]], on="lead_id", how="inner")
        joined["event_ts"] = pd.to_datetime(joined["event_ts"])
        joined["assignment_ts"] = pd.to_datetime(joined["assignment_ts"])
        joined = joined[joined["event_ts"] < joined["assignment_ts"]]

        # возраст события в часах на момент назначения
        joined["age_h"] = (
            joined["assignment_ts"] - joined["event_ts"]
        ).dt.total_seconds() / 3600

        by_lead = joined.groupby("lead_id")
        features = pd.DataFrame({
            "ev_n": by_lead.size(),  # активность лида за всю доступную историю
            "ev_recency_h": by_lead["age_h"].min(),       # свежесть последнего события (теплый/холодный лид)
            "ev_span_h": by_lead["age_h"].max() - by_lead["age_h"].min(), # расстояние между событиями (всплеск интереса/долгий интерес)
            "ev_slot_nuniq": by_lead["src_slot"].nunique(),  # через сколько разных слотов показа лид взаимодействовал
            "ev_ctx_nuniq": by_lead["ctx_seq"].nunique(),    # в скольких разных контекстах показа лид встречал объявления
            "ev_type_nuniq": by_lead["event_type"].nunique(),  # разнообразие типов действий, совершенных лидом (1-5)
        })

        # дальше оконные счетчики --- сколько раз лид сделал какое-либо действие
        # это больше повлияло на score
        features = features.join(self._window_features(joined), how="left")

        # _window_names - это колонки, порожденные _window_features
        # пропуск == событий не было => должны заполнить нулем
        window_names = self._window_names()
        features[window_names] = features[window_names].fillna(0)

        # лиды без событий до назначения (20 из 13694) получают NaN по всем колонкам
        return features.reindex(X["lead_id"].values).set_index(X.index)

    def _window_names(self):
        """Имена всех счётчиков, для которых пропуск означает 0, а не отсутствие истории
        """
        return (
            [f"ev_{event_type}_{window}d"
             for window in EVENT_WINDOWS for event_type in EVENT_TYPES]
            + [f"ev_hi_intent_{window}d" for window in EVENT_WINDOWS]
            + [f"ev_n_{window}d" for window in EVENT_WINDOWS]
            + [f"ev_ctx_{ctx}" for ctx in CTX_VALUES]
        )

    def _window_features(self, joined):
        """Счётчики по окнам, типам событий и контекстам - одним проходом по логу.

        Задача - получить счетчики - сколько событий типа Т попало в окно W до назначения лида
        """
        
        # каждому событию проставляется сумма флагов по группе - получаем счетчик
        # событие возрастом 0.5 дня получает w1=1, w3=1, w7=1, а возрастом 5 дней — w1=0, w3=0, w7=1
        joined = joined.copy()
        joined["age_d"] = joined["age_h"] / 24
        window_flags = [f"w{window}" for window in EVENT_WINDOWS]
        for window in EVENT_WINDOWS:
            joined[f"w{window}"] = (joined["age_d"] <= window).astype("int8")

        # счётчики (тип события x окно): суммируем флаги окон в одном groupby
        counts = (
            joined.groupby(["lead_id", "event_type"])[window_flags].sum()
            .unstack(fill_value=0)
            # тип может не встретиться в срезе - фиксируем состав колонок явно,
            # иначе набор признаков разойдётся между fit и transform
            .reindex(columns=pd.MultiIndex.from_product([window_flags, EVENT_TYPES]),
                     fill_value=0)
        )
        counts.columns = [f"ev_{event_type}_{flag[1:]}d" for flag, event_type in counts.columns]

        # hi_intent = сумма действий с намерением (см. в начале файла HIGH_INTENT_TYPES) - favorite/chat_open/call_click
        # считаем тоже по окнам
        high_intent = (
            joined[joined["event_type"].isin(HIGH_INTENT_TYPES)]
            .groupby("lead_id")[window_flags].sum()
        )
        high_intent.columns = [f"ev_hi_intent_{flag[1:]}d" for flag in high_intent.columns]

        # totals - сколько всего событий (любого типа) попало в окно W до назначения лида
        totals = joined.groupby("lead_id")[window_flags].sum()
        totals.columns = [f"ev_n_{flag[1:]}d" for flag in totals.columns]

        # контексты показа: сколько раз лид видел объявления в каждом контексте (c01..c08) - без окон
        ctx_counts = (
            joined.groupby(["lead_id", "ctx_seq"]).size().unstack(fill_value=0)
            .reindex(columns=CTX_VALUES, fill_value=0)
            .add_prefix("ev_ctx_")
        )

        return counts.join([high_intent, totals, ctx_counts], how="outer")


class Pipeline:
    """sklearn-обвязка, делегирует fit/predict_proba/predict

    sklearn_preprocess=False отключает ColumnTransformer и отдаёт модели сырой DataFrame, нужно для Catboost

    ranker=True для CatBoostRanker: ему нужен group_id (= день назначения), которого
    sklearn-пайплайн не умеет прокидывать, и он отдаёт скоры вместо вероятностей
    """

    def __init__(self, events=None, model=None, sklearn_preprocess=True, baseline=False,
                 ranker=False, **preprocessor_kwargs):
        self.events = events
        self.sklearn_preprocess = sklearn_preprocess
        self.baseline = baseline
        self.ranker = ranker
        self.model = model if model is not None else LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
        self.preprocessor_kwargs = preprocessor_kwargs
        self.pipeline_ = self._build()

    def _build(self):
        steps = []
        if self.baseline:
            steps.append(("drop_non_feature", FunctionTransformer(
                lambda X: X.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in X.columns])
            )))
        else:
            steps.append(("features", Preprocessor(events=self.events, **self.preprocessor_kwargs)))


        if self.sklearn_preprocess:
            numeric_preprocessor = SklearnPipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            categorical_preprocessor = SklearnPipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ])

            # колонки выбираем по dtype, а не жестким списком: после добавления
            # признаков из events заранее заданные списки колонок устарели бы
            steps.append(("preprocessor", ColumnTransformer(transformers=[
                ("num", numeric_preprocessor, make_column_selector(dtype_include=np.number)),
                ("cat", categorical_preprocessor, make_column_selector(dtype_include=object)),
            ], remainder="drop")))

        steps.append(("model", self.model))
        return SklearnPipeline(steps=steps)

    def fit(self, X, y, eval_set=None):
        if self.ranker:
            return self._fit_ranker(X, y, eval_set)

        if eval_set is None:
            self.pipeline_.fit(X, y)
            return self

        eval_X, eval_y = eval_set
        # eval_X надо вручную прогнать через шаги пайплайна
        *pre_steps, (_, model) = self.pipeline_.steps
        Xt, eval_Xt = X, eval_X
        for _, step in pre_steps:
            Xt = step.fit_transform(Xt, y)
            eval_Xt = step.transform(eval_Xt)
        model.fit(Xt, y, eval_set=[(eval_Xt, eval_y)])
        return self

    def _fit_ranker(self, X, y, eval_set=None):
        """Ранжирующий лосс учится порядку ВНУТРИ группы, а группа тут - день назначения:
        Daily AP считается по дням, так что группа метрики и группа обучения совпадают."""
        from catboost import Pool  # локально, чтобы модуль импортировался без catboost

        *pre_steps, (_, model) = self.pipeline_.steps

        def to_pool(raw, y_part, fit_steps):
            features = raw
            for _, step in pre_steps:
                features = step.fit_transform(features, y_part) if fit_steps else step.transform(features)
            groups = self._groups(raw)
            # catboost требует, чтобы строки одной группы шли подряд
            order = np.argsort(groups, kind="stable")
            return Pool(features.iloc[order], np.asarray(y_part)[order],
                        group_id=groups[order], cat_features=CATEGORICAL_COLUMNS)

        train_pool = to_pool(X, y, fit_steps=True)
        eval_pool = to_pool(*eval_set, fit_steps=False) if eval_set is not None else None
        model.fit(train_pool, eval_set=eval_pool)
        return self

    @staticmethod
    def _groups(X):
        return pd.to_datetime(X["assignment_date"]).dt.date.astype(str).values

    def predict_proba(self, X):
        if not self.ranker:
            return self.pipeline_.predict_proba(X)

        # ранкер выдаёт произвольные скоры, а не вероятности. Для Daily AP важен только
        # порядок, но формат сабмита требует [0, 1] - min-max порядок сохраняет
        scores = self.predict(X)
        low, high = scores.min(), scores.max()
        normalized = (scores - low) / (high - low) if high > low else np.zeros_like(scores)
        return np.column_stack([1.0 - normalized, normalized])

    def predict(self, X):
        if not self.ranker:
            return self.pipeline_.predict(X)

        features = X
        for _, step in self.pipeline_.steps[:-1]:
            features = step.transform(features)
        return self.model.predict(features)
