from __future__ import annotations

import io
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.inspection import (
    PartialDependenceDisplay,
    permutation_importance,
)
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Explainable AI | BI Copilot",
    page_icon="🔍",
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
        opacity: 0.94;
        max-width: 980px;
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
        <h1>Explainable Artificial Intelligence</h1>
        <p>
            Understand global feature importance, permutation importance,
            SHAP effects, individual predictions and partial dependence
            without treating model associations as causal relationships.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------

def get_model_components(
    model: Any,
) -> tuple[Any, Any]:
    if isinstance(model, Pipeline):
        preprocessor = model.named_steps.get(
            "preprocessor"
        )
        estimator = model.named_steps.get("model")

        if estimator is None:
            estimator = model.steps[-1][1]

        return preprocessor, estimator

    return None, model


def get_processed_features(
    model: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    preprocessor, _ = get_model_components(model)

    if preprocessor is None:
        processed = np.asarray(X)

        return processed, list(X.columns)

    processed = preprocessor.transform(X)
    processed = np.asarray(processed)

    try:
        feature_names = (
            preprocessor.get_feature_names_out().tolist()
        )
    except Exception:
        feature_names = [
            f"feature_{index}"
            for index in range(processed.shape[1])
        ]

    return processed, feature_names


def get_builtin_importance(
    estimator: Any,
    feature_names: list[str],
) -> pd.DataFrame | None:
    importance_values = None

    if hasattr(estimator, "feature_importances_"):
        importance_values = np.asarray(
            estimator.feature_importances_
        )

    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)

        if coefficients.ndim == 1:
            importance_values = np.abs(coefficients)
        else:
            importance_values = np.mean(
                np.abs(coefficients),
                axis=0,
            )

    if importance_values is None:
        return None

    if len(importance_values) != len(feature_names):
        return None

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    result = result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    return result


def grouped_feature_name(
    transformed_feature: str,
    original_columns: list[str],
) -> str:
    name = transformed_feature

    if "__" in name:
        name = name.split("__", 1)[1]

    exact_matches = [
        column
        for column in original_columns
        if name == column
    ]

    if exact_matches:
        return exact_matches[0]

    candidate_matches = [
        column
        for column in original_columns
        if name.startswith(f"{column}_")
    ]

    if candidate_matches:
        return max(
            candidate_matches,
            key=len,
        )

    return name


def group_transformed_importance(
    importance_df: pd.DataFrame,
    original_columns: list[str],
    importance_column: str,
) -> pd.DataFrame:
    grouped = importance_df.copy()

    grouped["original_feature"] = grouped[
        "feature"
    ].apply(
        lambda value: grouped_feature_name(
            value,
            original_columns,
        )
    )

    grouped = (
        grouped.groupby(
            "original_feature",
            as_index=False,
        )[importance_column]
        .sum()
        .sort_values(
            importance_column,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return grouped


def extract_shap_values(
    shap_result: Any,
    problem_type: str,
) -> np.ndarray:
    if isinstance(shap_result, list):
        if (
            problem_type == "classification"
            and len(shap_result) > 1
        ):
            values = np.asarray(shap_result[1])
        else:
            values = np.asarray(shap_result[0])
    elif hasattr(shap_result, "values"):
        values = np.asarray(shap_result.values)
    else:
        values = np.asarray(shap_result)

    if values.ndim == 3:
        if problem_type == "classification":
            class_index = 1 if values.shape[2] > 1 else 0
            values = values[:, :, class_index]
        else:
            values = values[:, :, 0]

    if values.ndim == 1:
        values = values.reshape(1, -1)

    return values


def dataframe_to_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------
# Retrieve trained model
# ---------------------------------------------------------

active_model = st.session_state.get("active_model")
active_model_name = st.session_state.get(
    "active_model_name"
)
X_train = st.session_state.get("X_train")
X_test = st.session_state.get("X_test")
y_test = st.session_state.get("y_test")
problem_type = st.session_state.get("problem_type")


if (
    active_model is None
    or not isinstance(X_train, pd.DataFrame)
    or not isinstance(X_test, pd.DataFrame)
    or y_test is None
):
    st.warning(
        "No trained supervised model is available. Open the "
        "**Supervised ML** page, train the models and then return here."
    )
    st.stop()


preprocessor, underlying_estimator = get_model_components(
    active_model
)


st.markdown(
    f"""
    <div class="status-success">
        <b>Active model:</b> {active_model_name}
        &nbsp; | &nbsp;
        <b>Problem:</b> {str(problem_type).title()}
        &nbsp; | &nbsp;
        <b>Training rows:</b> {len(X_train):,}
        &nbsp; | &nbsp;
        <b>Testing rows:</b> {len(X_test):,}
    </div>
    """,
    unsafe_allow_html=True,
)


processed_test, transformed_feature_names = (
    get_processed_features(
        active_model,
        X_test,
    )
)


summary_one, summary_two, summary_three, summary_four = st.columns(4)

summary_one.metric(
    "Model",
    active_model_name,
)
summary_two.metric(
    "Underlying algorithm",
    type(underlying_estimator).__name__,
)
summary_three.metric(
    "Original features",
    X_test.shape[1],
)
summary_four.metric(
    "Transformed features",
    processed_test.shape[1],
)


# ---------------------------------------------------------
# Built-in model importance
# ---------------------------------------------------------

st.subheader("1. Model-based feature importance")

builtin_importance = get_builtin_importance(
    underlying_estimator,
    transformed_feature_names,
)


if builtin_importance is not None:
    top_feature_count = st.slider(
        "Features displayed",
        min_value=5,
        max_value=min(
            30,
            len(builtin_importance),
        ),
        value=min(
            15,
            len(builtin_importance),
        ),
        key="builtin_top_features",
    )

    builtin_top = builtin_importance.head(
        top_feature_count
    )

    builtin_chart = px.bar(
        builtin_top.sort_values(
            "importance",
            ascending=True,
        ),
        x="importance",
        y="feature",
        orientation="h",
        title="Model-based transformed feature importance",
        labels={
            "importance": "Importance",
            "feature": "Feature",
        },
        color="importance",
        color_continuous_scale="Blues",
    )

    builtin_chart.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
    )

    st.plotly_chart(
        builtin_chart,
        width="stretch",
    )

    grouped_builtin = group_transformed_importance(
        builtin_importance,
        list(X_test.columns),
        "importance",
    )

    left_table, right_table = st.columns(2)

    with left_table:
        st.write("**Transformed features**")

        st.dataframe(
            builtin_importance.round(6),
            width="stretch",
            hide_index=True,
        )

    with right_table:
        st.write("**Grouped original features**")

        st.dataframe(
            grouped_builtin.round(6),
            width="stretch",
            hide_index=True,
        )

    st.session_state.builtin_feature_importance = (
        builtin_importance
    )
    st.session_state.grouped_builtin_importance = (
        grouped_builtin
    )

else:
    st.info(
        "This model does not expose direct feature importance or "
        "coefficient values. Permutation importance can still explain it."
    )


# ---------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------

st.subheader("2. Model-independent permutation importance")

st.markdown(
    """
    <div class="status-info">
        Permutation importance measures how much model performance decreases
        when one original feature is randomly shuffled. Larger decreases
        indicate stronger predictive importance.
    </div>
    """,
    unsafe_allow_html=True,
)


permutation_repeats = st.slider(
    "Permutation repetitions",
    min_value=3,
    max_value=30,
    value=10,
    step=1,
)


if st.button(
    "Calculate permutation importance",
    type="primary",
    width="stretch",
):
    with st.spinner(
        "Calculating permutation importance..."
    ):
        scoring = (
            "f1_macro"
            if problem_type == "classification"
            else "r2"
        )

        try:
            permutation_result = permutation_importance(
                estimator=active_model,
                X=X_test,
                y=y_test,
                scoring=scoring,
                n_repeats=permutation_repeats,
                random_state=42,
                n_jobs=-1,
            )

            permutation_df = pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "importance_mean": (
                        permutation_result.importances_mean
                    ),
                    "importance_std": (
                        permutation_result.importances_std
                    ),
                }
            )

            permutation_df = permutation_df.sort_values(
                "importance_mean",
                ascending=False,
            ).reset_index(drop=True)

            st.session_state.permutation_importance_df = (
                permutation_df
            )

        except Exception as error:
            st.error(
                f"Permutation importance failed: {error}"
            )


permutation_df = st.session_state.get(
    "permutation_importance_df"
)


if (
    isinstance(permutation_df, pd.DataFrame)
    and not permutation_df.empty
):
    permutation_chart = px.bar(
        permutation_df.sort_values(
            "importance_mean",
            ascending=True,
        ),
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std",
        title="Permutation importance on the holdout set",
        labels={
            "importance_mean": "Mean performance decrease",
            "feature": "Original feature",
        },
        color="importance_mean",
        color_continuous_scale="Blues",
    )

    permutation_chart.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
    )

    st.plotly_chart(
        permutation_chart,
        width="stretch",
    )

    st.dataframe(
        permutation_df.round(6),
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------

st.subheader("3. SHAP global explanations")

st.caption(
    "SHAP estimates how each transformed feature changes the model output "
    "relative to its baseline prediction."
)


shap_sample_size = st.slider(
    "SHAP sample size",
    min_value=20,
    max_value=min(500, len(X_test)),
    value=min(100, len(X_test)),
    step=10 if len(X_test) >= 30 else 1,
)


if st.button(
    "Generate SHAP explanations",
    width="stretch",
):
    try:
        import shap

        with st.spinner(
            "Generating SHAP explanations..."
        ):
            shap_X = X_test.iloc[
                :shap_sample_size
            ].copy()

            shap_processed, shap_feature_names = (
                get_processed_features(
                    active_model,
                    shap_X,
                )
            )

            background_size = min(
                100,
                len(X_train),
            )

            background_raw = X_train.sample(
                n=background_size,
                random_state=42,
            )

            background_processed, _ = (
                get_processed_features(
                    active_model,
                    background_raw,
                )
            )

            estimator_name = type(
                underlying_estimator
            ).__name__.lower()

            if (
                hasattr(
                    underlying_estimator,
                    "feature_importances_",
                )
                and "gradientboosting" not in estimator_name
                and "histgradient" not in estimator_name
            ):
                explainer = shap.TreeExplainer(
                    underlying_estimator
                )

                raw_shap_values = explainer.shap_values(
                    shap_processed
                )

            elif hasattr(
                underlying_estimator,
                "coef_",
            ):
                explainer = shap.LinearExplainer(
                    underlying_estimator,
                    background_processed,
                )

                raw_shap_values = explainer(
                    shap_processed
                )

            else:
                prediction_function = (
                    underlying_estimator.predict_proba
                    if (
                        problem_type == "classification"
                        and hasattr(
                            underlying_estimator,
                            "predict_proba",
                        )
                    )
                    else underlying_estimator.predict
                )

                explainer = shap.Explainer(
                    prediction_function,
                    background_processed,
                )

                raw_shap_values = explainer(
                    shap_processed
                )

            shap_values = extract_shap_values(
                raw_shap_values,
                problem_type,
            )

            mean_absolute_shap = np.mean(
                np.abs(shap_values),
                axis=0,
            )

            mean_signed_shap = np.mean(
                shap_values,
                axis=0,
            )

            shap_importance_df = pd.DataFrame(
                {
                    "feature": shap_feature_names,
                    "mean_absolute_shap": (
                        mean_absolute_shap
                    ),
                    "mean_signed_shap": (
                        mean_signed_shap
                    ),
                }
            )

            shap_importance_df = (
                shap_importance_df.sort_values(
                    "mean_absolute_shap",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            grouped_shap_df = (
                group_transformed_importance(
                    shap_importance_df,
                    list(X_test.columns),
                    "mean_absolute_shap",
                )
            )

            st.session_state.shap_values = (
                shap_values
            )
            st.session_state.shap_processed_features = (
                shap_processed
            )
            st.session_state.shap_feature_names = (
                shap_feature_names
            )
            st.session_state.shap_sample_raw = shap_X
            st.session_state.shap_importance_df = (
                shap_importance_df
            )
            st.session_state.grouped_shap_df = (
                grouped_shap_df
            )

    except ImportError:
        st.error(
            "SHAP is not installed. Run: pip install shap"
        )

    except Exception as error:
        st.error(
            f"SHAP explanation could not be generated: {error}"
        )


shap_importance_df = st.session_state.get(
    "shap_importance_df"
)
grouped_shap_df = st.session_state.get(
    "grouped_shap_df"
)


if (
    isinstance(shap_importance_df, pd.DataFrame)
    and not shap_importance_df.empty
):
    shap_chart = px.bar(
        shap_importance_df.head(20).sort_values(
            "mean_absolute_shap",
            ascending=True,
        ),
        x="mean_absolute_shap",
        y="feature",
        orientation="h",
        title="Global SHAP importance",
        labels={
            "mean_absolute_shap": "Mean absolute SHAP value",
            "feature": "Transformed feature",
        },
        color="mean_absolute_shap",
        color_continuous_scale="Blues",
    )

    shap_chart.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
    )

    st.plotly_chart(
        shap_chart,
        width="stretch",
    )

    shap_table_one, shap_table_two = st.columns(2)

    with shap_table_one:
        st.write("**Transformed SHAP importance**")

        st.dataframe(
            shap_importance_df.round(6),
            width="stretch",
            hide_index=True,
        )

    with shap_table_two:
        st.write("**Grouped original features**")

        st.dataframe(
            grouped_shap_df.round(6),
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------
# Individual prediction explanation
# ---------------------------------------------------------

st.subheader("4. Individual prediction explanation")

selected_row_number = st.number_input(
    "Select test row",
    min_value=0,
    max_value=max(
        0,
        len(X_test) - 1,
    ),
    value=0,
    step=1,
)


selected_row_number = int(selected_row_number)
selected_row = X_test.iloc[
    [selected_row_number]
].copy()

actual_value = pd.Series(y_test).iloc[
    selected_row_number
]

prediction = active_model.predict(
    selected_row
)[0]


individual_one, individual_two, individual_three = st.columns(3)

individual_one.metric(
    "Actual value",
    str(actual_value),
)
individual_two.metric(
    "Model prediction",
    str(prediction),
)


prediction_probability = None

if (
    problem_type == "classification"
    and hasattr(active_model, "predict_proba")
):
    probabilities = active_model.predict_proba(
        selected_row
    )[0]

    predicted_class_index = list(
        active_model.classes_
    ).index(prediction)

    prediction_probability = probabilities[
        predicted_class_index
    ]

    individual_three.metric(
        "Prediction confidence",
        f"{prediction_probability:.2%}",
    )

else:
    if problem_type == "regression":
        residual = float(
            actual_value - prediction
        )

        individual_three.metric(
            "Residual",
            f"{residual:,.4f}",
        )
    else:
        individual_three.metric(
            "Prediction confidence",
            "Not available",
        )


st.write("**Original input values**")

st.dataframe(
    selected_row,
    width="stretch",
    hide_index=True,
)


stored_shap_values = st.session_state.get(
    "shap_values"
)
stored_shap_features = st.session_state.get(
    "shap_processed_features"
)
stored_shap_names = st.session_state.get(
    "shap_feature_names"
)


if (
    isinstance(stored_shap_values, np.ndarray)
    and stored_shap_values.ndim == 2
    and selected_row_number < len(stored_shap_values)
):
    local_shap_values = stored_shap_values[
        selected_row_number
    ]

    local_feature_values = stored_shap_features[
        selected_row_number
    ]

    local_explanation = pd.DataFrame(
        {
            "feature": stored_shap_names,
            "feature_value": local_feature_values,
            "shap_value": local_shap_values,
            "absolute_shap": np.abs(
                local_shap_values
            ),
        }
    )

    local_explanation = (
        local_explanation.sort_values(
            "absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    local_chart_data = (
        local_explanation.head(15).copy()
    )

    local_chart_data["effect"] = np.where(
        local_chart_data["shap_value"] >= 0,
        "Increases model output",
        "Decreases model output",
    )

    local_chart = px.bar(
        local_chart_data.sort_values(
            "shap_value",
            ascending=True,
        ),
        x="shap_value",
        y="feature",
        orientation="h",
        color="effect",
        color_discrete_map={
            "Increases model output": "#2563eb",
            "Decreases model output": "#dc2626",
        },
        title=f"Local SHAP explanation for test row {selected_row_number}",
        labels={
            "shap_value": "SHAP effect",
            "feature": "Feature",
        },
    )

    local_chart.update_layout(
        template="plotly_white",
    )

    st.plotly_chart(
        local_chart,
        width="stretch",
    )

    st.dataframe(
        local_explanation.round(6),
        width="stretch",
        hide_index=True,
    )

else:
    st.info(
        "Generate SHAP explanations above to view the local feature "
        "contributions for this row."
    )


# ---------------------------------------------------------
# Partial dependence
# ---------------------------------------------------------

st.subheader("5. Partial dependence")

numeric_columns = X_train.select_dtypes(
    include=np.number
).columns.tolist()


if numeric_columns:
    selected_pdp_features = st.multiselect(
        "Numeric features for partial dependence",
        options=numeric_columns,
        default=numeric_columns[: min(3, len(numeric_columns))],
        max_selections=4,
    )

    if st.button(
        "Generate partial dependence plots",
        width="stretch",
    ):
        if not selected_pdp_features:
            st.warning(
                "Select at least one numeric feature."
            )
        else:
            try:
                X_partial = X_train.copy()

                # Convert integer numeric columns to floating point.
                # This prevents scikit-learn partial dependence errors.
                for column in numeric_columns:
                    X_partial[column] = pd.to_numeric(
                        X_partial[column],
                        errors="coerce",
                    ).astype(float)

                figure, axes = plt.subplots(
                    nrows=len(selected_pdp_features),
                    ncols=1,
                    figsize=(
                        11,
                        4 * len(selected_pdp_features),
                    ),
                    squeeze=False,
                )

                PartialDependenceDisplay.from_estimator(
                    estimator=active_model,
                    X=X_partial,
                    features=selected_pdp_features,
                    feature_names=list(X_partial.columns),
                    grid_resolution=30,
                    ax=axes[:, 0],
                    n_jobs=-1,
                )

                plt.tight_layout()

                st.pyplot(
                    figure,
                    clear_figure=True,
                )

                plt.close(figure)

            except Exception as error:
                st.error(
                    f"Partial dependence failed: {error}"
                )

else:
    st.info(
        "Partial dependence requires at least one numeric predictor."
    )


# ---------------------------------------------------------
# Downloads
# ---------------------------------------------------------

st.subheader("6. Download explanations")

download_columns = st.columns(3)

with download_columns[0]:
    if isinstance(
        builtin_importance,
        pd.DataFrame,
    ):
        st.download_button(
            "Download model importance",
            data=dataframe_to_csv_bytes(
                builtin_importance
            ),
            file_name="model_feature_importance.csv",
            mime="text/csv",
            width="stretch",
        )

with download_columns[1]:
    if isinstance(
        permutation_df,
        pd.DataFrame,
    ):
        st.download_button(
            "Download permutation importance",
            data=dataframe_to_csv_bytes(
                permutation_df
            ),
            file_name="permutation_importance.csv",
            mime="text/csv",
            width="stretch",
        )

with download_columns[2]:
    if isinstance(
        shap_importance_df,
        pd.DataFrame,
    ):
        st.download_button(
            "Download SHAP importance",
            data=dataframe_to_csv_bytes(
                shap_importance_df
            ),
            file_name="global_shap_importance.csv",
            mime="text/csv",
            width="stretch",
        )


st.warning(
    "Feature importance, SHAP values and partial dependence describe "
    "model associations. They do not prove that a feature causes the "
    "predicted outcome."
)