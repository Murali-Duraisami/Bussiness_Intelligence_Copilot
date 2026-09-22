from __future__ import annotations

import io
import json
import time
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Supervised ML | BI Copilot",
    page_icon="🤖",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .page-header {
        padding: 1.8rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        color: white;
        background: linear-gradient(
            120deg,
            #172554 0%,
            #1d4ed8 58%,
            #0ea5e9 100%
        );
        box-shadow: 0 16px 40px rgba(29, 78, 216, 0.18);
    }

    .page-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 750;
    }

    .page-header p {
        margin-top: 0.65rem;
        margin-bottom: 0;
        font-size: 1rem;
        opacity: 0.93;
        max-width: 920px;
    }

    .section-card {
        padding: 1.1rem 1.25rem;
        border: 1px solid #dbe5f1;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.045);
        margin-bottom: 1rem;
    }

    .status-success {
        padding: 0.9rem 1rem;
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    .status-info {
        padding: 0.9rem 1rem;
        background: #eff6ff;
        color: #1e3a8a;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #dbe5f1;
        background: white;
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="page-header">
        <h1>Supervised Machine Learning</h1>
        <p>
            Select a target feature, compare multiple classification or
            regression algorithms, evaluate cross-validation performance,
            inspect holdout results and save the best complete prediction
            pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def get_active_dataframe() -> pd.DataFrame | None:
    clean_df = st.session_state.get("clean_df")
    raw_df = st.session_state.get("raw_df")

    if isinstance(clean_df, pd.DataFrame) and not clean_df.empty:
        return clean_df.copy()

    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        return raw_df.copy()

    return None


def infer_problem_type(target: pd.Series) -> str:
    non_null_target = target.dropna()

    if non_null_target.empty:
        return "Classification"

    if (
        pd.api.types.is_object_dtype(non_null_target)
        or pd.api.types.is_bool_dtype(non_null_target)
        or isinstance(non_null_target.dtype, pd.CategoricalDtype)
    ):
        return "Classification"

    unique_count = non_null_target.nunique()
    threshold = max(20, int(len(non_null_target) * 0.05))

    if unique_count <= threshold:
        return "Classification"

    return "Regression"


def detect_possible_id_columns(
    dataframe: pd.DataFrame,
    target_column: str,
) -> list[str]:
    possible_ids: list[str] = []

    for column in dataframe.columns:
        if column == target_column:
            continue

        normalized_name = column.lower().strip()

        name_looks_like_id = (
            normalized_name == "id"
            or normalized_name.endswith("_id")
            or normalized_name.startswith("id_")
            or "identifier" in normalized_name
        )

        unique_ratio = dataframe[column].nunique(dropna=True) / max(
            len(dataframe),
            1,
        )

        if name_looks_like_id or unique_ratio >= 0.98:
            possible_ids.append(column)

    return possible_ids


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )


def classification_models(random_state: int) -> dict[str, Any]:
    model_collection: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=180,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
        "Support Vector Machine": SVC(
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "K-Nearest Neighbours": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "ANN MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=random_state,
        ),
    }

    try:
        from xgboost import XGBClassifier

        model_collection["XGBoost"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        model_collection["LightGBM"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            verbosity=-1,
            random_state=random_state,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        model_collection["CatBoost"] = CatBoostClassifier(
            iterations=250,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_seed=random_state,
        )
    except ImportError:
        pass

    return model_collection


def regression_models(random_state: int) -> dict[str, Any]:
    model_collection: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(
            alpha=0.01,
            max_iter=5000,
            random_state=random_state,
        ),
        "Elastic Net": ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=4,
            random_state=random_state,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.07,
            max_iter=180,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
        "Support Vector Regression": SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
        ),
        "K-Nearest Neighbours": KNeighborsRegressor(
            n_neighbors=7,
            weights="distance",
        ),
        "ANN MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=700,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=random_state,
        ),
    }

    try:
        from xgboost import XGBRegressor

        model_collection["XGBoost"] = XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor

        model_collection["LightGBM"] = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            verbosity=-1,
            random_state=random_state,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor

        model_collection["CatBoost"] = CatBoostRegressor(
            iterations=250,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_seed=random_state,
        )
    except ImportError:
        pass

    return model_collection


def create_pipeline(
    X: pd.DataFrame,
    estimator: Any,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(X),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def classification_holdout_metrics(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, np.ndarray | None]:
    predictions = model.predict(X_test)

    metrics = {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_balanced_accuracy": balanced_accuracy_score(
            y_test,
            predictions,
        ),
        "test_precision_macro": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "test_recall_macro": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "test_f1_macro": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "test_f1_weighted": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    probabilities = None
    unique_classes = np.unique(y_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)

        try:
            metrics["test_log_loss"] = log_loss(
                y_test,
                probabilities,
                labels=model.classes_,
            )
        except Exception:
            metrics["test_log_loss"] = np.nan

        try:
            if len(unique_classes) == 2:
                metrics["test_roc_auc"] = roc_auc_score(
                    y_test,
                    probabilities[:, 1],
                )
            else:
                metrics["test_roc_auc"] = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                    labels=model.classes_,
                )
        except Exception:
            metrics["test_roc_auc"] = np.nan
    else:
        metrics["test_log_loss"] = np.nan
        metrics["test_roc_auc"] = np.nan

    return metrics, predictions, probabilities


def regression_holdout_metrics(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = model.predict(X_test)

    metrics = {
        "test_mae": mean_absolute_error(y_test, predictions),
        "test_rmse": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "test_r2": r2_score(y_test, predictions),
    }

    return metrics, predictions


def serialize_model(model: Pipeline) -> bytes:
    memory_file = io.BytesIO()
    joblib.dump(model, memory_file)
    memory_file.seek(0)
    return memory_file.getvalue()


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------
# Load active data
# ---------------------------------------------------------

df = get_active_dataframe()

if df is None:
    st.warning(
        "No active dataset was found. Return to the Home page and upload "
        "a dataset or select the demonstration dataset."
    )
    st.stop()


st.markdown(
    f"""
    <div class="status-success">
        <b>Active dataset:</b>
        {st.session_state.get("dataset_name", "Current dataset")}
        &nbsp; | &nbsp;
        <b>{df.shape[0]:,}</b> rows
        &nbsp; | &nbsp;
        <b>{df.shape[1]:,}</b> columns
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

st.subheader("1. Configure the machine-learning experiment")

configuration_left, configuration_middle, configuration_right = st.columns(
    [1.15, 1, 1]
)

with configuration_left:
    previous_target = st.session_state.get("target_column")

    if previous_target in df.columns:
        default_target_index = list(df.columns).index(previous_target)
    else:
        default_target_index = len(df.columns) - 1

    target_column = st.selectbox(
        "Target feature",
        options=list(df.columns),
        index=default_target_index,
        help="The model will learn to predict this column.",
    )

with configuration_middle:
    detected_problem = infer_problem_type(df[target_column])

    problem_type = st.selectbox(
        "Problem type",
        options=["Classification", "Regression"],
        index=0 if detected_problem == "Classification" else 1,
        help=(
            f"Automatically detected as {detected_problem}. "
            "You can override the selection."
        ),
    )

with configuration_right:
    test_size = st.slider(
        "Test-set percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )


possible_id_columns = detect_possible_id_columns(
    df,
    target_column,
)

id_columns = st.multiselect(
    "Columns to exclude from model training",
    options=[
        column
        for column in df.columns
        if column != target_column
    ],
    default=possible_id_columns,
    help=(
        "Identifiers such as customer_id usually should not be used as "
        "predictive features."
    ),
)


random_state_column, folds_column = st.columns(2)

with random_state_column:
    random_state = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

with folds_column:
    requested_folds = st.selectbox(
        "Cross-validation folds",
        options=[3, 5, 7, 10],
        index=1,
    )


feature_columns = [
    column
    for column in df.columns
    if column != target_column and column not in id_columns
]

if not feature_columns:
    st.error("No predictor columns remain after exclusions.")
    st.stop()


model_options = (
    classification_models(int(random_state))
    if problem_type == "Classification"
    else regression_models(int(random_state))
)


if problem_type == "Classification":
    recommended_models = [
        model
        for model in [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "ANN MLP",
        ]
        if model in model_options
    ]
else:
    recommended_models = [
        model
        for model in [
            "Ridge Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "ANN MLP",
        ]
        if model in model_options
    ]


selected_models = st.multiselect(
    "Models to compare",
    options=list(model_options.keys()),
    default=recommended_models,
    help=(
        "ANN MLP is included as the artificial neural-network model. "
        "Selecting every model can increase training time."
    ),
)


with st.expander("Review experiment variables"):
    numeric_features = (
        df[feature_columns]
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_features = [
        column
        for column in feature_columns
        if column not in numeric_features
    ]

    review_one, review_two, review_three = st.columns(3)

    review_one.metric("Predictor features", len(feature_columns))
    review_two.metric("Numeric features", len(numeric_features))
    review_three.metric(
        "Categorical features",
        len(categorical_features),
    )

    st.write("**Predictors:**", feature_columns)

    if id_columns:
        st.write("**Excluded columns:**", id_columns)


# ---------------------------------------------------------
# Target overview
# ---------------------------------------------------------

st.subheader("2. Target overview")

target_missing = int(df[target_column].isna().sum())
target_unique = int(df[target_column].nunique(dropna=True))

target_metric_one, target_metric_two, target_metric_three = st.columns(3)

target_metric_one.metric(
    "Available target values",
    f"{len(df) - target_missing:,}",
)
target_metric_two.metric("Missing target values", f"{target_missing:,}")
target_metric_three.metric("Unique target values", f"{target_unique:,}")


if problem_type == "Classification":
    target_counts = (
        df[target_column]
        .astype("string")
        .fillna("Missing")
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Count")
    )

    target_counts["Percentage"] = (
        target_counts["Count"] / target_counts["Count"].sum() * 100
    )

    figure = px.bar(
        target_counts,
        x="Class",
        y="Count",
        color="Class",
        text=target_counts["Percentage"].map(
            lambda value: f"{value:.1f}%"
        ),
        title="Target class distribution",
    )

    figure.update_layout(
        showlegend=False,
        template="plotly_white",
    )

    st.plotly_chart(figure, width="stretch")

    if target_counts["Percentage"].min() < 15:
        st.warning(
            "The target is imbalanced. Balanced accuracy and macro F1 "
            "will be emphasized during evaluation."
        )

else:
    numeric_target = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    figure = px.histogram(
        x=numeric_target.dropna(),
        nbins=30,
        title=f"Distribution of {target_column}",
        labels={"x": target_column},
    )

    figure.update_layout(template="plotly_white")
    st.plotly_chart(figure, width="stretch")


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

st.subheader("3. Train and compare models")

st.markdown(
    """
    <div class="status-info">
        Cross-validation is performed only on the training partition.
        The final test partition remains separate until each fitted model
        is evaluated.
    </div>
    """,
    unsafe_allow_html=True,
)


train_button = st.button(
    "Train selected models",
    type="primary",
    width="stretch",
    disabled=len(selected_models) == 0,
)


if train_button:
    working_df = df[
        feature_columns + [target_column]
    ].copy()

    working_df = working_df.dropna(subset=[target_column])

    X = working_df[feature_columns].copy()
    y = working_df[target_column].copy()

    if problem_type == "Regression":
        y = pd.to_numeric(y, errors="coerce")
        valid_target_mask = y.notna()
        X = X.loc[valid_target_mask].copy()
        y = y.loc[valid_target_mask].copy()

    if len(X) < 30:
        st.error(
            "At least 30 usable records are recommended for this "
            "training workflow."
        )
        st.stop()

    if problem_type == "Classification":
        class_counts = y.value_counts()

        if len(class_counts) < 2:
            st.error(
                "Classification requires at least two target classes."
            )
            st.stop()

        if class_counts.min() < 2:
            st.error(
                "Every target class requires at least two records."
            )
            st.stop()

        stratify_values = y if class_counts.min() >= 2 else None
    else:
        stratify_values = None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=int(random_state),
            stratify=stratify_values,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=int(random_state),
        )

    if problem_type == "Classification":
        minimum_training_class = int(
            y_train.value_counts().min()
        )

        actual_folds = min(
            int(requested_folds),
            minimum_training_class,
        )

        if actual_folds < 2:
            st.error(
                "The training data does not contain enough examples per "
                "class for cross-validation."
            )
            st.stop()

        cross_validator = StratifiedKFold(
            n_splits=actual_folds,
            shuffle=True,
            random_state=int(random_state),
        )

        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
        }

    else:
        actual_folds = min(
            int(requested_folds),
            max(2, len(X_train) // 10),
        )

        cross_validator = KFold(
            n_splits=actual_folds,
            shuffle=True,
            random_state=int(random_state),
        )

        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }

    progress_bar = st.progress(0)
    progress_message = st.empty()

    trained_models: dict[str, Pipeline] = {}
    result_rows: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    failed_models: list[dict[str, str]] = []

    for model_index, model_name in enumerate(selected_models):
        progress_message.info(
            f"Training {model_name} "
            f"({model_index + 1} of {len(selected_models)})..."
        )

        estimator = clone(model_options[model_name])
        pipeline = create_pipeline(X_train, estimator)

        started_at = time.perf_counter()

        try:
            cv_result = cross_validate(
                estimator=pipeline,
                X=X_train,
                y=y_train,
                cv=cross_validator,
                scoring=scoring,
                return_train_score=True,
                n_jobs=1,
                error_score="raise",
            )

            pipeline.fit(X_train, y_train)
            training_seconds = time.perf_counter() - started_at

            result_row: dict[str, Any] = {
                "model": model_name,
                "cv_folds": actual_folds,
                "training_seconds": training_seconds,
            }

            if problem_type == "Classification":
                result_row.update(
                    {
                        "cv_accuracy_mean": np.mean(
                            cv_result["test_accuracy"]
                        ),
                        "cv_accuracy_std": np.std(
                            cv_result["test_accuracy"]
                        ),
                        "cv_balanced_accuracy_mean": np.mean(
                            cv_result["test_balanced_accuracy"]
                        ),
                        "cv_f1_macro_mean": np.mean(
                            cv_result["test_f1_macro"]
                        ),
                        "cv_f1_macro_std": np.std(
                            cv_result["test_f1_macro"]
                        ),
                        "cv_f1_weighted_mean": np.mean(
                            cv_result["test_f1_weighted"]
                        ),
                        "train_f1_macro_mean": np.mean(
                            cv_result["train_f1_macro"]
                        ),
                    }
                )

                result_row["train_validation_gap"] = (
                    result_row["train_f1_macro_mean"]
                    - result_row["cv_f1_macro_mean"]
                )

                holdout_metrics, predictions, probabilities = (
                    classification_holdout_metrics(
                        pipeline,
                        X_test,
                        y_test,
                    )
                )

                result_row.update(holdout_metrics)

                prediction_data = X_test.reset_index(drop=True).copy()
                prediction_data["actual"] = y_test.reset_index(drop=True)
                prediction_data["prediction"] = predictions

                if (
                    probabilities is not None
                    and probabilities.shape[1] == 2
                ):
                    prediction_data["probability_class_1"] = (
                        probabilities[:, 1]
                    )

            else:
                result_row.update(
                    {
                        "cv_r2_mean": np.mean(
                            cv_result["test_r2"]
                        ),
                        "cv_r2_std": np.std(
                            cv_result["test_r2"]
                        ),
                        "cv_mae_mean": -np.mean(
                            cv_result["test_mae"]
                        ),
                        "cv_rmse_mean": -np.mean(
                            cv_result["test_rmse"]
                        ),
                        "train_r2_mean": np.mean(
                            cv_result["train_r2"]
                        ),
                    }
                )

                result_row["train_validation_gap"] = (
                    result_row["train_r2_mean"]
                    - result_row["cv_r2_mean"]
                )

                holdout_metrics, predictions = (
                    regression_holdout_metrics(
                        pipeline,
                        X_test,
                        y_test,
                    )
                )

                result_row.update(holdout_metrics)

                prediction_data = X_test.reset_index(drop=True).copy()
                prediction_data["actual"] = y_test.reset_index(drop=True)
                prediction_data["prediction"] = predictions
                prediction_data["residual"] = (
                    prediction_data["actual"]
                    - prediction_data["prediction"]
                )

            trained_models[model_name] = pipeline
            result_rows.append(result_row)
            prediction_frames[model_name] = prediction_data

        except Exception as error:
            failed_models.append(
                {
                    "model": model_name,
                    "error": str(error),
                }
            )

        progress_bar.progress(
            (model_index + 1) / len(selected_models)
        )

    progress_message.empty()
    progress_bar.empty()

    if not result_rows:
        st.error("All selected models failed during training.")

        if failed_models:
            st.dataframe(
                pd.DataFrame(failed_models),
                width="stretch",
                hide_index=True,
            )

        st.stop()

    results_df = pd.DataFrame(result_rows)

    if problem_type == "Classification":
        results_df = results_df.sort_values(
            by="cv_f1_macro_mean",
            ascending=False,
        ).reset_index(drop=True)
    else:
        results_df = results_df.sort_values(
            by="cv_r2_mean",
            ascending=False,
        ).reset_index(drop=True)

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_predictions = prediction_frames[best_model_name]

    st.session_state.target_column = target_column
    st.session_state.problem_type = problem_type.lower()
    st.session_state.feature_columns = feature_columns
    st.session_state.trained_models = trained_models
    st.session_state.active_model = best_model
    st.session_state.active_model_name = best_model_name
    st.session_state.model_results = results_df
    st.session_state.model_predictions = best_predictions
    st.session_state.all_model_predictions = prediction_frames
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.model_failures = failed_models
    st.session_state.cv_folds = actual_folds

    st.success(
        f"Training completed. Validation-selected model: "
        f"{best_model_name}"
    )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

results_df = st.session_state.get("model_results")
active_model = st.session_state.get("active_model")
active_model_name = st.session_state.get("active_model_name")
predictions_df = st.session_state.get("model_predictions")
stored_problem_type = st.session_state.get("problem_type")


if (
    isinstance(results_df, pd.DataFrame)
    and not results_df.empty
    and active_model is not None
):
    st.divider()
    st.subheader("4. Model leaderboard")

    display_results = results_df.copy()

    numeric_result_columns = display_results.select_dtypes(
        include=np.number
    ).columns

    display_results[numeric_result_columns] = display_results[
        numeric_result_columns
    ].round(4)

    st.dataframe(
        display_results,
        width="stretch",
        hide_index=True,
    )

    if stored_problem_type == "classification":
        chart_df = results_df.sort_values(
            "cv_f1_macro_mean",
            ascending=True,
        )

        chart = px.bar(
            chart_df,
            x="cv_f1_macro_mean",
            y="model",
            orientation="h",
            error_x="cv_f1_macro_std",
            title="Cross-validation macro F1 comparison",
            labels={
                "cv_f1_macro_mean": "Mean macro F1",
                "model": "Model",
            },
            color="cv_f1_macro_mean",
            color_continuous_scale="Blues",
        )

    else:
        chart_df = results_df.sort_values(
            "cv_r2_mean",
            ascending=True,
        )

        chart = px.bar(
            chart_df,
            x="cv_r2_mean",
            y="model",
            orientation="h",
            error_x="cv_r2_std",
            title="Cross-validation R² comparison",
            labels={
                "cv_r2_mean": "Mean R²",
                "model": "Model",
            },
            color="cv_r2_mean",
            color_continuous_scale="Blues",
        )

    chart.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
    )

    st.plotly_chart(chart, width="stretch")

    st.subheader("5. Best-model holdout evaluation")

    best_row = results_df.loc[
        results_df["model"] == active_model_name
    ].iloc[0]

    if stored_problem_type == "classification":
        metric_one, metric_two, metric_three, metric_four = st.columns(4)

        metric_one.metric(
            "Selected model",
            active_model_name,
        )
        metric_two.metric(
            "Test accuracy",
            f"{best_row['test_accuracy']:.2%}",
        )
        metric_three.metric(
            "Macro F1",
            f"{best_row['test_f1_macro']:.4f}",
        )

        roc_auc_value = best_row.get("test_roc_auc", np.nan)

        metric_four.metric(
            "ROC-AUC",
            (
                f"{roc_auc_value:.4f}"
                if pd.notna(roc_auc_value)
                else "Not available"
            ),
        )

        y_test = st.session_state.get("y_test")

        if y_test is not None and isinstance(
            predictions_df,
            pd.DataFrame,
        ):
            labels = np.unique(
                np.concatenate(
                    [
                        np.asarray(y_test),
                        predictions_df["prediction"].to_numpy(),
                    ]
                )
            )

            matrix = confusion_matrix(
                y_test,
                predictions_df["prediction"],
                labels=labels,
            )

            confusion_df = pd.DataFrame(
                matrix,
                index=[
                    f"Actual {label}" for label in labels
                ],
                columns=[
                    f"Predicted {label}" for label in labels
                ],
            )

            st.write("**Confusion matrix**")
            st.dataframe(
                confusion_df,
                width="stretch",
            )

    else:
        metric_one, metric_two, metric_three, metric_four = st.columns(4)

        metric_one.metric(
            "Selected model",
            active_model_name,
        )
        metric_two.metric(
            "Test MAE",
            f"{best_row['test_mae']:,.4f}",
        )
        metric_three.metric(
            "Test RMSE",
            f"{best_row['test_rmse']:,.4f}",
        )
        metric_four.metric(
            "Test R²",
            f"{best_row['test_r2']:.4f}",
        )

        if isinstance(predictions_df, pd.DataFrame):
            actual_vs_predicted = px.scatter(
                predictions_df,
                x="actual",
                y="prediction",
                title="Actual versus predicted values",
                labels={
                    "actual": "Actual value",
                    "prediction": "Predicted value",
                },
            )

            minimum_value = min(
                predictions_df["actual"].min(),
                predictions_df["prediction"].min(),
            )
            maximum_value = max(
                predictions_df["actual"].max(),
                predictions_df["prediction"].max(),
            )

            actual_vs_predicted.add_shape(
                type="line",
                x0=minimum_value,
                y0=minimum_value,
                x1=maximum_value,
                y1=maximum_value,
                line={
                    "color": "#dc2626",
                    "dash": "dash",
                },
            )

            actual_vs_predicted.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                actual_vs_predicted,
                width="stretch",
            )

    with st.expander("View holdout predictions"):
        st.dataframe(
            predictions_df,
            width="stretch",
            hide_index=True,
        )

    failures = st.session_state.get("model_failures", [])

    if failures:
        with st.expander("Models that could not be trained"):
            st.dataframe(
                pd.DataFrame(failures),
                width="stretch",
                hide_index=True,
            )

    st.subheader("6. Download trained outputs")

    download_one, download_two, download_three = st.columns(3)

    with download_one:
        st.download_button(
            label="Download best model",
            data=serialize_model(active_model),
            file_name="best_supervised_model.joblib",
            mime="application/octet-stream",
            width="stretch",
        )

    with download_two:
        st.download_button(
            label="Download leaderboard",
            data=dataframe_to_csv_bytes(results_df),
            file_name="supervised_model_results.csv",
            mime="text/csv",
            width="stretch",
        )

    with download_three:
        st.download_button(
            label="Download predictions",
            data=dataframe_to_csv_bytes(predictions_df),
            file_name="supervised_test_predictions.csv",
            mime="text/csv",
            width="stretch",
        )

    model_summary = {
        "target_column": st.session_state.get("target_column"),
        "problem_type": stored_problem_type,
        "selected_model": active_model_name,
        "feature_columns": st.session_state.get(
            "feature_columns",
            [],
        ),
        "cross_validation_folds": st.session_state.get(
            "cv_folds"
        ),
        "selection_rule": (
            "Highest cross-validation macro F1"
            if stored_problem_type == "classification"
            else "Highest cross-validation R-squared"
        ),
    }

    st.download_button(
        label="Download model summary",
        data=json.dumps(
            model_summary,
            indent=2,
            default=str,
        ),
        file_name="supervised_model_summary.json",
        mime="application/json",
        width="stretch",
    )

else:
    st.info(
        "Configure the experiment and select **Train selected models** "
        "to generate the model leaderboard."
    )