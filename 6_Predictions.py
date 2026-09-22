from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Predictions | BI Copilot",
    page_icon="🎯",
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
        <h1>Prediction Centre</h1>
        <p>
            Generate individual predictions, inspect class probabilities,
            upload new datasets for batch scoring and export prediction
            results using the validation-selected model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------

def load_uploaded_data(
    uploaded_file: Any,
) -> pd.DataFrame:
    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()

    if extension == ".csv":
        return pd.read_csv(uploaded_file)

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)

    if extension == ".json":
        try:
            return pd.read_json(uploaded_file)
        except ValueError:
            uploaded_file.seek(0)

            return pd.read_json(
                uploaded_file,
                lines=True,
            )

    if extension in {".parquet", ".pq"}:
        return pd.read_parquet(uploaded_file)

    if extension in {".tsv", ".txt"}:
        return pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
        )

    raise ValueError(
        f"Unsupported file extension: {extension}"
    )


def prepare_batch_features(
    dataframe: pd.DataFrame,
    required_features: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    missing_features = [
        feature
        for feature in required_features
        if feature not in dataframe.columns
    ]

    extra_features = [
        column
        for column in dataframe.columns
        if column not in required_features
    ]

    if missing_features:
        return dataframe, missing_features, extra_features

    prepared = dataframe[
        required_features
    ].copy()

    return prepared, missing_features, extra_features


def prediction_results(
    model: Any,
    features: pd.DataFrame,
    problem_type: str,
) -> pd.DataFrame:
    output = features.copy()
    predictions = model.predict(features)

    output["prediction"] = predictions

    if (
        problem_type == "classification"
        and hasattr(model, "predict_proba")
    ):
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)

        for class_index, class_value in enumerate(classes):
            probability_column = (
                f"probability_{str(class_value)}"
            )

            output[probability_column] = probabilities[
                :,
                class_index,
            ]

        output["prediction_confidence"] = (
            probabilities.max(axis=1)
        )

    return output


def csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def excel_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    memory_file = io.BytesIO()

    with pd.ExcelWriter(
        memory_file,
        engine="openpyxl",
    ) as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Predictions",
        )

    memory_file.seek(0)
    return memory_file.getvalue()


def safe_default_number(
    values: pd.Series,
) -> float:
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric_values.empty:
        return 0.0

    return float(numeric_values.median())


def numeric_step(
    values: pd.Series,
) -> float:
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric_values.empty:
        return 0.1

    spread = float(
        numeric_values.max()
        - numeric_values.min()
    )

    if spread == 0:
        return 0.1

    calculated_step = spread / 100

    return max(
        calculated_step,
        0.0001,
    )


# ---------------------------------------------------------
# Retrieve active model
# ---------------------------------------------------------

active_model = st.session_state.get("active_model")
active_model_name = st.session_state.get(
    "active_model_name"
)
problem_type = st.session_state.get("problem_type")
target_column = st.session_state.get(
    "target_column"
)
feature_columns = st.session_state.get(
    "feature_columns",
    [],
)
X_train = st.session_state.get("X_train")


if (
    active_model is None
    or not feature_columns
    or not isinstance(X_train, pd.DataFrame)
):
    st.warning(
        "No trained model is available. Open the **Supervised ML** page, "
        "train the models and then return to this page."
    )
    st.stop()


st.markdown(
    f"""
    <div class="status-success">
        <b>Active model:</b> {active_model_name}
        &nbsp; | &nbsp;
        <b>Problem:</b> {str(problem_type).title()}
        &nbsp; | &nbsp;
        <b>Target:</b> {target_column}
        &nbsp; | &nbsp;
        <b>Required features:</b> {len(feature_columns)}
    </div>
    """,
    unsafe_allow_html=True,
)


summary_one, summary_two, summary_three = st.columns(3)

summary_one.metric(
    "Prediction model",
    active_model_name,
)
summary_two.metric(
    "Target feature",
    target_column,
)
summary_three.metric(
    "Input features",
    len(feature_columns),
)


# ---------------------------------------------------------
# Prediction tabs
# ---------------------------------------------------------

manual_tab, batch_tab, history_tab = st.tabs(
    [
        "Individual prediction",
        "Batch prediction",
        "Prediction history",
    ]
)


# ---------------------------------------------------------
# Individual prediction
# ---------------------------------------------------------

with manual_tab:
    st.subheader("Individual prediction")

    st.markdown(
        """
        <div class="status-info">
            Enter one record below. The values are passed through the same
            preprocessing pipeline used during model training.
        </div>
        """,
        unsafe_allow_html=True,
    )

    manual_values: dict[str, Any] = {}

    input_columns = st.columns(2)

    for feature_index, feature in enumerate(
        feature_columns
    ):
        container = input_columns[
            feature_index % 2
        ]

        training_values = X_train[feature]

        with container:
            if pd.api.types.is_bool_dtype(
                training_values
            ):
                unique_values = (
                    training_values.dropna()
                    .unique()
                    .tolist()
                )

                if not unique_values:
                    unique_values = [False, True]

                manual_values[feature] = st.selectbox(
                    feature,
                    options=unique_values,
                    key=f"manual_{feature}",
                )

            elif pd.api.types.is_numeric_dtype(
                training_values
            ):
                default_value = safe_default_number(
                    training_values
                )

                step_value = numeric_step(
                    training_values
                )

                if pd.api.types.is_integer_dtype(
                    training_values
                ):
                    manual_values[feature] = (
                        st.number_input(
                            feature,
                            value=int(round(default_value)),
                            step=1,
                            key=f"manual_{feature}",
                        )
                    )

                else:
                    manual_values[feature] = (
                        st.number_input(
                            feature,
                            value=float(default_value),
                            step=float(step_value),
                            format="%.6f",
                            key=f"manual_{feature}",
                        )
                    )

            elif pd.api.types.is_datetime64_any_dtype(
                training_values
            ):
                non_null_dates = (
                    training_values.dropna()
                )

                if non_null_dates.empty:
                    default_date = pd.Timestamp.today()
                else:
                    default_date = pd.Timestamp(
                        non_null_dates.median()
                    )

                selected_date = st.date_input(
                    feature,
                    value=default_date.date(),
                    key=f"manual_{feature}",
                )

                manual_values[feature] = pd.Timestamp(
                    selected_date
                )

            else:
                category_values = (
                    training_values.dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                category_values = sorted(
                    category_values
                )

                if len(category_values) <= 100:
                    manual_values[feature] = (
                        st.selectbox(
                            feature,
                            options=category_values,
                            key=f"manual_{feature}",
                        )
                    )
                else:
                    default_text = (
                        category_values[0]
                        if category_values
                        else ""
                    )

                    manual_values[feature] = (
                        st.text_input(
                            feature,
                            value=default_text,
                            key=f"manual_{feature}",
                        )
                    )

    if st.button(
        "Generate individual prediction",
        type="primary",
        width="stretch",
    ):
        try:
            manual_input = pd.DataFrame(
                [manual_values],
                columns=feature_columns,
            )

            individual_result = prediction_results(
                active_model,
                manual_input,
                problem_type,
            )

            st.session_state.latest_individual_prediction = (
                individual_result
            )

            prediction_history = (
                st.session_state.get(
                    "prediction_history",
                    [],
                )
            )

            history_record = individual_result.copy()
            history_record.insert(
                0,
                "predicted_at",
                pd.Timestamp.now(),
            )

            prediction_history.append(
                history_record
            )

            st.session_state.prediction_history = (
                prediction_history[-100:]
            )

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )

    individual_result = st.session_state.get(
        "latest_individual_prediction"
    )

    if (
        isinstance(individual_result, pd.DataFrame)
        and not individual_result.empty
    ):
        predicted_value = individual_result.iloc[0][
            "prediction"
        ]

        if problem_type == "classification":
            result_one, result_two = st.columns(2)

            result_one.metric(
                "Predicted class",
                str(predicted_value),
            )

            confidence_value = individual_result.iloc[
                0
            ].get(
                "prediction_confidence",
                np.nan,
            )

            result_two.metric(
                "Prediction confidence",
                (
                    f"{confidence_value:.2%}"
                    if pd.notna(confidence_value)
                    else "Not available"
                ),
            )

            probability_columns = [
                column
                for column in individual_result.columns
                if column.startswith(
                    "probability_"
                )
            ]

            if probability_columns:
                probability_data = pd.DataFrame(
                    {
                        "class": [
                            column.replace(
                                "probability_",
                                "",
                                1,
                            )
                            for column in probability_columns
                        ],
                        "probability": [
                            individual_result.iloc[0][
                                column
                            ]
                            for column in probability_columns
                        ],
                    }
                )

                probability_chart = px.bar(
                    probability_data,
                    x="class",
                    y="probability",
                    color="class",
                    text=probability_data[
                        "probability"
                    ].map(
                        lambda value: f"{value:.1%}"
                    ),
                    title="Predicted class probabilities",
                    labels={
                        "class": "Target class",
                        "probability": "Probability",
                    },
                )

                probability_chart.update_layout(
                    template="plotly_white",
                    showlegend=False,
                    yaxis_tickformat=".0%",
                )

                st.plotly_chart(
                    probability_chart,
                    width="stretch",
                )

        else:
            st.metric(
                "Predicted value",
                f"{float(predicted_value):,.6f}",
            )

        st.write("**Prediction record**")

        st.dataframe(
            individual_result,
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------

with batch_tab:
    st.subheader("Batch prediction")

    uploaded_batch_file = st.file_uploader(
        "Upload new records",
        type=[
            "csv",
            "tsv",
            "txt",
            "xlsx",
            "xls",
            "json",
            "parquet",
            "pq",
        ],
        key="batch_prediction_upload",
        help=(
            "The uploaded dataset must contain all features used during "
            "model training."
        ),
    )

    st.write("**Required input columns**")

    required_columns_df = pd.DataFrame(
        {
            "position": range(
                1,
                len(feature_columns) + 1,
            ),
            "required_feature": feature_columns,
            "training_dtype": [
                str(X_train[column].dtype)
                for column in feature_columns
            ],
        }
    )

    st.dataframe(
        required_columns_df,
        width="stretch",
        hide_index=True,
    )

    if uploaded_batch_file is not None:
        try:
            batch_df = load_uploaded_data(
                uploaded_batch_file
            )

            st.success(
                f"Loaded {batch_df.shape[0]:,} rows and "
                f"{batch_df.shape[1]:,} columns."
            )

            st.write("**Uploaded data preview**")

            st.dataframe(
                batch_df.head(20),
                width="stretch",
                hide_index=True,
            )

            (
                batch_features,
                missing_features,
                extra_features,
            ) = prepare_batch_features(
                batch_df,
                feature_columns,
            )

            if missing_features:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_features)
                )

            else:
                st.success(
                    "All required model features are available."
                )

                if extra_features:
                    st.info(
                        "Additional columns will remain in the "
                        "downloaded result but will not be used by "
                        "the prediction model: "
                        + ", ".join(extra_features)
                    )

                if st.button(
                    "Run batch prediction",
                    type="primary",
                    width="stretch",
                ):
                    with st.spinner(
                        "Generating batch predictions..."
                    ):
                        batch_prediction_only = (
                            prediction_results(
                                active_model,
                                batch_features,
                                problem_type,
                            )
                        )

                        final_batch_results = (
                            batch_df.copy()
                        )

                        prediction_output_columns = [
                            column
                            for column in (
                                batch_prediction_only.columns
                            )
                            if column not in feature_columns
                        ]

                        for output_column in (
                            prediction_output_columns
                        ):
                            final_batch_results[
                                output_column
                            ] = batch_prediction_only[
                                output_column
                            ].to_numpy()

                        st.session_state.batch_prediction_results = (
                            final_batch_results
                        )

        except Exception as error:
            st.error(
                f"Could not process the uploaded file: {error}"
            )

    batch_results = st.session_state.get(
        "batch_prediction_results"
    )

    if (
        isinstance(batch_results, pd.DataFrame)
        and not batch_results.empty
    ):
        st.write("**Batch prediction results**")

        batch_summary_one, batch_summary_two = (
            st.columns(2)
        )

        batch_summary_one.metric(
            "Scored records",
            f"{len(batch_results):,}",
        )

        if problem_type == "classification":
            predicted_distribution = (
                batch_results["prediction"]
                .astype(str)
                .value_counts()
                .rename_axis("predicted_class")
                .reset_index(name="records")
            )

            batch_summary_two.metric(
                "Predicted classes",
                predicted_distribution[
                    "predicted_class"
                ].nunique(),
            )

            distribution_chart = px.bar(
                predicted_distribution,
                x="predicted_class",
                y="records",
                color="predicted_class",
                title="Batch prediction distribution",
                labels={
                    "predicted_class": "Predicted class",
                    "records": "Records",
                },
            )

            distribution_chart.update_layout(
                template="plotly_white",
                showlegend=False,
            )

            st.plotly_chart(
                distribution_chart,
                width="stretch",
            )

        else:
            batch_summary_two.metric(
                "Average prediction",
                (
                    f"{batch_results['prediction'].mean():,.4f}"
                ),
            )

            prediction_distribution = px.histogram(
                batch_results,
                x="prediction",
                nbins=30,
                title="Predicted value distribution",
            )

            prediction_distribution.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                prediction_distribution,
                width="stretch",
            )

        st.dataframe(
            batch_results,
            width="stretch",
            hide_index=True,
        )

        download_csv, download_excel = st.columns(2)

        with download_csv:
            st.download_button(
                "Download predictions as CSV",
                data=csv_bytes(batch_results),
                file_name="batch_predictions.csv",
                mime="text/csv",
                width="stretch",
            )

        with download_excel:
            st.download_button(
                "Download predictions as Excel",
                data=excel_bytes(batch_results),
                file_name="batch_predictions.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                width="stretch",
            )


# ---------------------------------------------------------
# Prediction history
# ---------------------------------------------------------

with history_tab:
    st.subheader("Individual prediction history")

    prediction_history = st.session_state.get(
        "prediction_history",
        [],
    )

    if prediction_history:
        combined_history = pd.concat(
            prediction_history,
            ignore_index=True,
        )

        st.dataframe(
            combined_history,
            width="stretch",
            hide_index=True,
        )

        history_download, history_clear = st.columns(2)

        with history_download:
            st.download_button(
                "Download prediction history",
                data=csv_bytes(
                    combined_history
                ),
                file_name="prediction_history.csv",
                mime="text/csv",
                width="stretch",
            )

        with history_clear:
            if st.button(
                "Clear prediction history",
                width="stretch",
            ):
                st.session_state.prediction_history = []
                st.session_state.latest_individual_prediction = None
                st.rerun()

    else:
        st.info(
            "Individual predictions generated during this session "
            "will appear here."
        )


st.warning(
    "Predictions are statistical estimates and may be wrong. Validate "
    "the model with current real-world data before using its output for "
    "important business decisions."
)