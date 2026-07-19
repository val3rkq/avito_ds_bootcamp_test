# файл содержит пайплайн предобработки данных (feature engineering)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "target"
ID_COLUMNS = ["lead_id", "user_id"]
TIME_COLUMNS = ["assignment_ts", "assignment_date"]
NON_FEATURE_COLUMNS = ID_COLUMNS + TIME_COLUMNS + [TARGET, "split"]

WINDOW_ORDER = ["1d", "3d", "7d", "14d", "30d", "90d"]

# признаки из events.csv, у которых нашелся сигнал по correlation ratio с target
EVENT_TYPES = ["item_view", "search", "favorite", "chat_open", "call_click"]

# признаки, дублирующие другие (EDA1.2)
REDUNDANT_COLUMNS = ["is_weekend", "price_bucket"]

# категориальные признаки на выходе Preprocessor (для CatBoostClassifier)
CATEGORICAL_COLUMNS = ["lead_source", "call_center", "region",
                       "car_segment", "lead_channel", "user_tenure_bucket"]


class Preprocessor(BaseEstimator, TransformerMixin):
    """Feature engineering: строит признаки из events.csv и чистит дубликаты.

    Наследуемся от sklearn-базовых классов, чтобы объект можно было положить шагом
    в Pipeline: тогда fit вызывается на train-части каждого фолда отдельно, и утечка
    между фолдами исключена по построению.

    transform принимает СЫРОЙ DataFrame строк (с lead_id и assignment_ts), а не срез
    по feature_columns: признакам из events нужны обе эти колонки.
    """

    def __init__(self, events=None, use_events=True, drop_redundant=True):
        self.events = events
        self.use_events = use_events
        self.drop_redundant = drop_redundant

    def fit(self, X, y=None):
        # y не используется: все признаки строятся построчно из прошлого самой строки
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
        events = self.events[["lead_id", "event_ts", "event_type", "src_slot",
                              "ctx_seq", "item_price_log"]]
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
            "ev_n": by_lead.size(),
            "ev_recency_h": by_lead["age_h"].min(),       # свежесть последнего события
            "ev_span_h": by_lead["age_h"].max() - by_lead["age_h"].min(),
            "ev_slot_nuniq": by_lead["src_slot"].nunique(),
            "ev_ctx_nuniq": by_lead["ctx_seq"].nunique(),
            "ev_type_nuniq": by_lead["event_type"].nunique(),
            # std одного значения не определён; для лида с единственным событием
            # разброс цен физически нулевой, а NaN здесь означал бы "событий нет вовсе"
            "ev_price_std": by_lead["item_price_log"].std().fillna(0.0),
        })

        counts = (
            joined.groupby(["lead_id", "event_type"]).size().unstack(fill_value=0)
            .reindex(columns=EVENT_TYPES, fill_value=0)
            .add_prefix("ev_n_")
        )
        features = features.join(counts)

        # лиды без событий до назначения (0.07%) получают NaN
        return features.reindex(X["lead_id"].values).set_index(X.index)


class Pipeline:
    """sklearn-обвязка
    
    Делегирует fit/predict_proba/predict, поэтому объект подходит как drop-in
    для цикла по фолдам из main.ipynb.

    sklearn_preprocess=False отключает ColumnTransformer и отдаёт модели сырой DataFrame, нужно для Catboost
    """

    def __init__(self, events=None, model=None, sklearn_preprocess=True, baseline=False, **preprocessor_kwargs):
        self.events = events
        self.sklearn_preprocess = sklearn_preprocess
        self.baseline = baseline
        self.model = model if model is not None else LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
        self.preprocessor_kwargs = preprocessor_kwargs
        self.pipeline_ = self._build()

    def _build(self):
        steps = []
        if not self.baseline:
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

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(X)

    def predict(self, X):
        return self.pipeline_.predict(X)
