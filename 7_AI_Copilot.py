from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Copilot | BI Copilot",
    page_icon="💬",
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
        max-width: 950px;
    }

    .status-success {
        padding: 0.9rem 1rem;
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    .suggestion-card {
        padding: 0.9rem;
        border: 1px solid #dbe5f1;
        border-radius: 12px;
        background: #f8fafc;
        margin-bottom: 0.6rem;
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
        <h1>Natural Language AI Copilot</h1>
        <p>
            Ask questions about the active dataset, descriptive statistics,
            missing values, correlations, trained models, feature importance,
            customer segments and potential anomalies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Dataset access
# ---------------------------------------------------------

def get_active_dataframe() -> pd.DataFrame | None:
    clean_df = st.session_state.get("clean_df")
    raw_df = st.session_state.get("raw_df")

    if isinstance(clean_df, pd.DataFrame) and not clean_df.empty:
        return clean_df.copy()

    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        return raw_df.copy()

    return None


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
# Question-answer functions
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def detect_column(
    question: str,
    columns: list[str],
) -> str | None:
    normalized_question = normalize_text(question)

    # Check longer column names first.
    sorted_columns = sorted(
        columns,
        key=lambda value: len(str(value)),
        reverse=True,
    )

    for column in sorted_columns:
        column_text = str(column)
        normalized_column = normalize_text(column_text)

        pattern = (
            r"(?<![a-zA-Z0-9])"
            + re.escape(normalized_column)
            + r"(?![a-zA-Z0-9])"
        )

        if re.search(pattern, normalized_question):
            return column_text

    # Try matching tokens from underscore-based column names.
    for column in sorted_columns:
        tokens = [
            token
            for token in re.split(
                r"[_\s]+",
                str(column).lower(),
            )
            if len(token) >= 3
        ]

        if tokens and all(
            re.search(
                r"\b" + re.escape(token) + r"\b",
                normalized_question,
            )
            for token in tokens
        ):
            return str(column)

    return None


def format_number(value: Any) -> str:
    try:
        numeric_value = float(value)

        if np.isnan(numeric_value):
            return "not available"

        if abs(numeric_value) >= 1000:
            return f"{numeric_value:,.4f}"

        return f"{numeric_value:.4f}"

    except Exception:
        return str(value)


def total_missing_answer(dataframe: pd.DataFrame) -> str:
    missing_counts = dataframe.isna().sum()
    total_missing = int(missing_counts.sum())

    if total_missing == 0:
        return "The dataset has no missing values."

    affected_columns = missing_counts[
        missing_counts > 0
    ].sort_values(ascending=False)

    details = ", ".join(
        f"{column}={int(count)}"
        for column, count in affected_columns.items()
    )

    return (
        f"The dataset contains {total_missing:,} missing values. "
        f"Missing values by affected column: {details}."
    )


def column_missing_answer(
    dataframe: pd.DataFrame,
    column: str,
) -> str:
    missing_count = int(
        dataframe[column].isna().sum()
    )

    missing_percentage = (
        missing_count / len(dataframe) * 100
    )

    return (
        f"{column} contains {missing_count:,} missing values, "
        f"representing {missing_percentage:.2f}% of the dataset."
    )


def numeric_summary_answer(
    dataframe: pd.DataFrame,
    column: str,
) -> str:
    numeric_values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if numeric_values.empty:
        return (
            f"{column} does not contain usable numeric values."
        )

    return (
        f"For {column}: mean={format_number(numeric_values.mean())}, "
        f"median={format_number(numeric_values.median())}, "
        f"minimum={format_number(numeric_values.min())}, "
        f"maximum={format_number(numeric_values.max())}, "
        f"standard deviation={format_number(numeric_values.std())}."
    )


def distribution_answer(
    dataframe: pd.DataFrame,
    column: str,
) -> str:
    counts = (
        dataframe[column]
        .astype("string")
        .fillna("Missing")
        .value_counts()
        .head(10)
    )

    details = "; ".join(
        (
            f"{value}: {int(count):,} "
            f"({count / len(dataframe) * 100:.2f}%)"
        )
        for value, count in counts.items()
    )

    return (
        f"The most frequent values in {column} are: {details}."
    )


def correlation_answer(
    dataframe: pd.DataFrame,
    question: str,
) -> str:
    numeric_df = dataframe.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:
        return (
            "At least two numeric columns are required to calculate "
            "correlations."
        )

    mentioned_columns = []

    for column in numeric_df.columns:
        normalized_column = normalize_text(str(column))

        if re.search(
            r"\b"
            + re.escape(normalized_column)
            + r"\b",
            normalize_text(question),
        ):
            mentioned_columns.append(column)

    mentioned_columns = list(
        dict.fromkeys(mentioned_columns)
    )

    if len(mentioned_columns) >= 2:
        first_column = mentioned_columns[0]
        second_column = mentioned_columns[1]

        pearson_value = numeric_df[
            first_column
        ].corr(
            numeric_df[second_column],
            method="pearson",
        )

        spearman_value = numeric_df[
            first_column
        ].corr(
            numeric_df[second_column],
            method="spearman",
        )

        return (
            f"The Pearson correlation between {first_column} and "
            f"{second_column} is {pearson_value:.4f}. "
            f"The Spearman correlation is {spearman_value:.4f}. "
            "Correlation describes association and does not prove causation."
        )

    correlation_matrix = numeric_df.corr(
        method="spearman"
    )

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(
                correlation_matrix.shape,
                dtype=bool,
            ),
            k=1,
        )
    )

    correlation_pairs = (
        upper_triangle.stack()
        .rename("correlation")
        .reset_index()
    )

    if correlation_pairs.empty:
        return "No numeric correlation pairs are available."

    correlation_pairs["absolute_correlation"] = (
        correlation_pairs["correlation"].abs()
    )

    strongest = correlation_pairs.sort_values(
        "absolute_correlation",
        ascending=False,
    ).iloc[0]

    return (
        f"The strongest Spearman relationship is between "
        f"{strongest['level_0']} and {strongest['level_1']}, "
        f"with correlation {strongest['correlation']:.4f}. "
        "Correlation does not establish causation."
    )


def best_model_answer() -> str:
    active_model_name = st.session_state.get(
        "active_model_name"
    )

    if not active_model_name:
        return (
            "No supervised model has been trained in the current session."
        )

    problem_type = st.session_state.get(
        "problem_type",
        "unknown",
    )

    return (
        f"The validation-selected {problem_type} model is "
        f"{active_model_name}. Model selection was based on "
        "cross-validation performance."
    )


def model_performance_answer() -> str:
    results = st.session_state.get(
        "model_results"
    )
    active_model_name = st.session_state.get(
        "active_model_name"
    )
    problem_type = st.session_state.get(
        "problem_type"
    )

    if (
        not isinstance(results, pd.DataFrame)
        or results.empty
        or not active_model_name
    ):
        return (
            "Model performance is not available. Train a supervised model "
            "first."
        )

    selected_rows = results.loc[
        results["model"] == active_model_name
    ]

    if selected_rows.empty:
        return "The selected model metrics are unavailable."

    selected = selected_rows.iloc[0]

    if problem_type == "classification":
        accuracy = selected.get(
            "test_accuracy",
            np.nan,
        )

        macro_f1 = selected.get(
            "test_f1_macro",
            np.nan,
        )

        balanced_accuracy = selected.get(
            "test_balanced_accuracy",
            np.nan,
        )

        roc_auc = selected.get(
            "test_roc_auc",
            np.nan,
        )

        answer = (
            f"{active_model_name} achieved "
            f"{accuracy:.2%} test accuracy, "
            f"{balanced_accuracy:.4f} balanced accuracy and "
            f"{macro_f1:.4f} macro F1."
        )

        if pd.notna(roc_auc):
            answer += f" ROC-AUC was {roc_auc:.4f}."

        return answer

    mae = selected.get("test_mae", np.nan)
    rmse = selected.get("test_rmse", np.nan)
    r2 = selected.get("test_r2", np.nan)

    return (
        f"{active_model_name} achieved test MAE of "
        f"{mae:,.4f}, RMSE of {rmse:,.4f}, "
        f"and R-squared of {r2:.4f}."
    )


def feature_importance_answer() -> str:
    grouped_shap = st.session_state.get(
        "grouped_shap_df"
    )

    permutation_df = st.session_state.get(
        "permutation_importance_df"
    )

    grouped_builtin = st.session_state.get(
        "grouped_builtin_importance"
    )

    if (
        isinstance(grouped_shap, pd.DataFrame)
        and not grouped_shap.empty
    ):
        top_features = grouped_shap.head(5)[
            "original_feature"
        ].astype(str).tolist()

        return (
            "Based on grouped SHAP importance, the most influential "
            "features are: "
            + ", ".join(top_features)
            + "."
        )

    if (
        isinstance(permutation_df, pd.DataFrame)
        and not permutation_df.empty
    ):
        top_features = permutation_df.head(5)[
            "feature"
        ].astype(str).tolist()

        return (
            "Based on permutation importance, the most influential "
            "features are: "
            + ", ".join(top_features)
            + "."
        )

    if (
        isinstance(grouped_builtin, pd.DataFrame)
        and not grouped_builtin.empty
    ):
        top_features = grouped_builtin.head(5)[
            "original_feature"
        ].astype(str).tolist()

        return (
            "Based on model feature importance, the most influential "
            "features are: "
            + ", ".join(top_features)
            + "."
        )

    return (
        "Feature importance has not been generated. Open the Explainable "
        "AI page and calculate permutation importance or SHAP values."
    )


def clustering_answer() -> str:
    cluster_count = st.session_state.get(
        "selected_cluster_count"
    )

    evaluation = st.session_state.get(
        "kmeans_evaluation"
    )

    if cluster_count is None:
        return (
            "Clustering results are unavailable. Run the Unsupervised ML "
            "analysis first."
        )

    silhouette_value = np.nan

    if isinstance(evaluation, pd.DataFrame):
        selected_row = evaluation.loc[
            evaluation["number_of_clusters"]
            == cluster_count
        ]

        if not selected_row.empty:
            silhouette_value = selected_row.iloc[0].get(
                "silhouette_score",
                np.nan,
            )

    answer = (
        f"K-Means selected {cluster_count} clusters."
    )

    if pd.notna(silhouette_value):
        answer += (
            f" The silhouette score is "
            f"{silhouette_value:.4f}."
        )

        if silhouette_value < 0.25:
            answer += (
                " This is a weak clustering structure, so the groups "
                "overlap and require business validation."
            )
        elif silhouette_value < 0.50:
            answer += (
                " The clustering structure is moderate and should be "
                "reviewed before operational use."
            )
        else:
            answer += (
                " The clusters show reasonably strong separation."
            )

    return answer


def anomaly_answer() -> str:
    anomaly_data = st.session_state.get(
        "anomaly_data"
    )

    clustered_data = st.session_state.get(
        "clustered_data"
    )

    if not isinstance(anomaly_data, pd.DataFrame):
        return (
            "Anomaly results are unavailable. Run the Unsupervised ML "
            "analysis first."
        )

    anomaly_count = len(anomaly_data)

    if (
        isinstance(clustered_data, pd.DataFrame)
        and len(clustered_data) > 0
    ):
        anomaly_percentage = (
            anomaly_count / len(clustered_data) * 100
        )

        return (
            f"Isolation Forest identified {anomaly_count:,} potential "
            f"anomalies, representing {anomaly_percentage:.2f}% of "
            "the analyzed records."
        )

    return (
        f"Isolation Forest identified {anomaly_count:,} potential "
        "anomalies."
    )


def duplicate_answer(
    dataframe: pd.DataFrame,
) -> str:
    duplicate_count = int(
        dataframe.duplicated().sum()
    )

    duplicate_percentage = (
        duplicate_count / len(dataframe) * 100
    )

    return (
        f"The dataset contains {duplicate_count:,} duplicate rows, "
        f"representing {duplicate_percentage:.2f}% of all records."
    )


def answer_question(
    question: str,
    dataframe: pd.DataFrame,
) -> str:
    normalized_question = normalize_text(
        question
    )

    detected_column = detect_column(
        question,
        list(dataframe.columns),
    )

    # Dataset size
    if (
        "how many rows" in normalized_question
        or "number of rows" in normalized_question
        or "row count" in normalized_question
        or "records" in normalized_question
        and "how many" in normalized_question
    ):
        return (
            f"The dataset has {dataframe.shape[0]:,} rows."
        )

    if (
        "how many columns" in normalized_question
        or "number of columns" in normalized_question
        or "column count" in normalized_question
        or "features are in" in normalized_question
    ):
        return (
            f"The dataset has {dataframe.shape[1]:,} columns: "
            + ", ".join(
                map(str, dataframe.columns)
            )
            + "."
        )

    # Missing values
    if (
        "missing" in normalized_question
        or "null" in normalized_question
        or "none values" in normalized_question
    ):
        if detected_column:
            return column_missing_answer(
                dataframe,
                detected_column,
            )

        return total_missing_answer(dataframe)

    # Duplicates
    if (
        "duplicate" in normalized_question
        or "duplicated" in normalized_question
    ):
        return duplicate_answer(dataframe)

    # Model questions
    if (
        "best model" in normalized_question
        or "selected model" in normalized_question
        or "which model" in normalized_question
    ):
        return best_model_answer()

    if (
        "model accuracy" in normalized_question
        or "model performance" in normalized_question
        or "macro f1" in normalized_question
        or "roc auc" in normalized_question
        or "r squared" in normalized_question
        or "rmse" in normalized_question
    ):
        return model_performance_answer()

    if (
        "important feature" in normalized_question
        or "feature importance" in normalized_question
        or "most influential" in normalized_question
        or "shap" in normalized_question
    ):
        return feature_importance_answer()

    # Unsupervised questions
    if (
        "how many clusters" in normalized_question
        or "number of clusters" in normalized_question
        or "customer segments" in normalized_question
        or "silhouette" in normalized_question
    ):
        return clustering_answer()

    if (
        "anomaly" in normalized_question
        or "anomalies" in normalized_question
        or "outlier records" in normalized_question
    ):
        return anomaly_answer()

    # Correlation
    if (
        "correlation" in normalized_question
        or "relationship" in normalized_question
        or "associated" in normalized_question
    ):
        return correlation_answer(
            dataframe,
            question,
        )

    # Mean
    if (
        "average" in normalized_question
        or "mean" in normalized_question
    ):
        if detected_column:
            numeric_values = pd.to_numeric(
                dataframe[detected_column],
                errors="coerce",
            ).dropna()

            if numeric_values.empty:
                return (
                    f"{detected_column} is not numeric, so its arithmetic "
                    "mean cannot be calculated."
                )

            return (
                f"The mean of {detected_column} is "
                f"{format_number(numeric_values.mean())}."
            )

        return (
            "Please mention the exact column name whose average you want."
        )

    # Median
    if "median" in normalized_question:
        if detected_column:
            numeric_values = pd.to_numeric(
                dataframe[detected_column],
                errors="coerce",
            ).dropna()

            if numeric_values.empty:
                return (
                    f"{detected_column} is not numeric, so its median "
                    "cannot be calculated."
                )

            return (
                f"The median of {detected_column} is "
                f"{format_number(numeric_values.median())}."
            )

        return (
            "Please mention the exact column name whose median you want."
        )

    # Minimum
    if (
        "minimum" in normalized_question
        or "lowest" in normalized_question
        or "smallest" in normalized_question
    ):
        if detected_column:
            numeric_values = pd.to_numeric(
                dataframe[detected_column],
                errors="coerce",
            ).dropna()

            if numeric_values.empty:
                return (
                    f"{detected_column} does not contain numeric values."
                )

            return (
                f"The minimum value of {detected_column} is "
                f"{format_number(numeric_values.min())}."
            )

    # Maximum
    if (
        "maximum" in normalized_question
        or "highest" in normalized_question
        or "largest" in normalized_question
    ):
        if detected_column:
            numeric_values = pd.to_numeric(
                dataframe[detected_column],
                errors="coerce",
            ).dropna()

            if numeric_values.empty:
                return (
                    f"{detected_column} does not contain numeric values."
                )

            return (
                f"The maximum value of {detected_column} is "
                f"{format_number(numeric_values.max())}."
            )

    # Unique values
    if (
        "unique" in normalized_question
        or "distinct" in normalized_question
    ):
        if detected_column:
            unique_count = dataframe[
                detected_column
            ].nunique(dropna=True)

            sample_values = (
                dataframe[detected_column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()[:15]
            )

            return (
                f"{detected_column} has {unique_count:,} unique "
                f"non-missing values. Examples: "
                + ", ".join(sample_values)
                + "."
            )

    # Distribution
    if (
        "distribution" in normalized_question
        or "value counts" in normalized_question
        or "most common" in normalized_question
        or "frequency" in normalized_question
    ):
        if detected_column:
            return distribution_answer(
                dataframe,
                detected_column,
            )

        return (
            "Please mention the exact column name whose distribution "
            "you want."
        )

    # General column summary
    if detected_column:
        if pd.api.types.is_numeric_dtype(
            dataframe[detected_column]
        ):
            return numeric_summary_answer(
                dataframe,
                detected_column,
            )

        return distribution_answer(
            dataframe,
            detected_column,
        )

    return (
        "I can answer questions about rows, columns, missing values, "
        "duplicates, averages, medians, minimums, maximums, unique values, "
        "distributions, correlations, trained models, feature importance, "
        "clusters and anomalies. Mention an exact column name when relevant."
    )


# ---------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------

st.subheader("Suggested questions")

suggested_questions = [
    "How many rows are in the dataset?",
    "How many missing values are there?",
    "What is the average income?",
    "What is the distribution of region?",
    "What is the correlation between income and purchased?",
    "Which is the best model?",
    "What is the model accuracy?",
    "What are the most important features?",
    "How many clusters were created?",
    "How many anomalies were found?",
]


suggestion_columns = st.columns(2)

for question_index, suggested_question in enumerate(
    suggested_questions
):
    with suggestion_columns[
        question_index % 2
    ]:
        if st.button(
            suggested_question,
            key=f"suggestion_{question_index}",
            width="stretch",
        ):
            st.session_state.copilot_pending_question = (
                suggested_question
            )


# ---------------------------------------------------------
# Chat interface
# ---------------------------------------------------------

st.subheader("Ask the copilot")

if "copilot_messages" not in st.session_state:
    st.session_state.copilot_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello. Ask me a question about the active dataset, "
                "statistical results, trained models, clusters or anomalies."
            ),
        }
    ]


pending_question = st.session_state.pop(
    "copilot_pending_question",
    None,
)


for message in st.session_state.copilot_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_question = st.chat_input(
    "Ask a business intelligence question..."
)


question_to_process = (
    user_question
    if user_question
    else pending_question
)


if question_to_process:
    st.session_state.copilot_messages.append(
        {
            "role": "user",
            "content": question_to_process,
        }
    )

    answer = answer_question(
        question_to_process,
        df,
    )

    st.session_state.copilot_messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.session_state.latest_copilot_answer = answer

    st.rerun()


# ---------------------------------------------------------
# Conversation controls
# ---------------------------------------------------------

control_one, control_two = st.columns(2)

with control_one:
    if st.button(
        "Clear conversation",
        width="stretch",
    ):
        st.session_state.copilot_messages = [
            {
                "role": "assistant",
                "content": (
                    "Conversation cleared. Ask me a new question about "
                    "the active analysis."
                ),
            }
        ]

        st.rerun()

with control_two:
    conversation_lines = []

    for message in st.session_state.copilot_messages:
        role_name = message["role"].title()

        conversation_lines.append(
            f"{role_name}: {message['content']}"
        )

    conversation_text = "\n\n".join(
        conversation_lines
    )

    st.download_button(
        "Download conversation",
        data=conversation_text.encode("utf-8"),
        file_name="bi_copilot_conversation.txt",
        mime="text/plain",
        width="stretch",
    )


st.caption(
    "This copilot runs locally using deterministic analytical rules. "
    "It does not send the uploaded dataset to a paid external AI API."
)