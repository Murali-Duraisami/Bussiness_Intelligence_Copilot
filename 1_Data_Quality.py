from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.core import (
    clean,
    profile,
    quality_score,
)


st.set_page_config(
    page_title="Data Quality | BI Copilot",
    page_icon="🧹",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        .page-header {
            padding: 1.5rem 1.7rem;
            border-radius: 18px;
            color: white;
            background:
                linear-gradient(
                    120deg,
                    #172554,
                    #2563eb
                );
            margin-bottom: 1.3rem;
        }

        .page-title {
            font-size: 2rem;
            font-weight: 750;
        }

        .page-description {
            color: #dbeafe;
            margin-top: 0.4rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1rem;
            background: white;
        }

        .quality-good {
            padding: 1rem;
            border-radius: 12px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
        }

        .quality-warning {
            padding: 1rem;
            border-radius: 12px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="page-header">
        <div class="page-title">
            Data Quality and Cleaning
        </div>
        <div class="page-description">
            Profile the active dataset, inspect quality problems
            and create a cleaned dataset for analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if (
    "raw_df" not in st.session_state
    or st.session_state.raw_df is None
):
    st.warning(
        "No active dataset is available. Return to the "
        "Home page and upload a dataset or load the "
        "demonstration data."
    )

    st.stop()


df = st.session_state.raw_df.copy()

dataset_summary, column_summary = profile(df)
score = quality_score(df)


# =========================================================
# SUMMARY
# =========================================================

summary_columns = st.columns(6)

summary_columns[0].metric(
    "Quality score",
    f"{score:.1f}%",
)

summary_columns[1].metric(
    "Rows",
    f"{len(df):,}",
)

summary_columns[2].metric(
    "Columns",
    f"{df.shape[1]:,}",
)

summary_columns[3].metric(
    "Missing values",
    f"{df.isna().sum().sum():,}",
)

summary_columns[4].metric(
    "Duplicate rows",
    f"{df.duplicated().sum():,}",
)

summary_columns[5].metric(
    "Memory",
    f"{dataset_summary['memory_mb']:.3f} MB",
)


if score >= 90:
    st.markdown(
        """
        <div class="quality-good">
            The dataset has excellent structural quality.
            Continue reviewing statistical distributions,
            outliers and business validity.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <div class="quality-warning">
            Quality issues were detected. Review the tabs
            below and configure the cleaning options before
            model training.
        </div>
        """,
        unsafe_allow_html=True,
    )


tabs = st.tabs(
    [
        "Column profile",
        "Missing values",
        "Duplicates",
        "Outliers",
        "Cardinality",
        "Clean dataset",
    ]
)


# =========================================================
# COLUMN PROFILE
# =========================================================

with tabs[0]:
    st.subheader("Column-level profile")

    st.dataframe(
        column_summary,
        use_container_width=True,
        hide_index=True,
    )

    selected_column = st.selectbox(
        "Inspect a column",
        options=df.columns,
    )

    selected_series = df[selected_column]

    detail_columns = st.columns(4)

    detail_columns[0].metric(
        "Data type",
        str(selected_series.dtype),
    )

    detail_columns[1].metric(
        "Unique values",
        f"{selected_series.nunique(dropna=True):,}",
    )

    detail_columns[2].metric(
        "Missing",
        f"{selected_series.isna().sum():,}",
    )

    detail_columns[3].metric(
        "Missing percentage",
        f"{selected_series.isna().mean() * 100:.2f}%",
    )

    st.write("Most frequent values")

    frequency_df = (
        selected_series.value_counts(
            dropna=False
        )
        .head(20)
        .rename_axis("value")
        .reset_index(name="frequency")
    )

    st.dataframe(
        frequency_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# MISSING VALUES
# =========================================================

with tabs[1]:
    st.subheader("Missing-value analysis")

    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count":
            df.isna().sum().values,
        "missing_percentage":
            (
                df.isna().mean()
                * 100
            ).round(2).values,
    })

    missing_df = missing_df.sort_values(
        "missing_percentage",
        ascending=False,
    )

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
    )

    columns_with_missing = missing_df.loc[
        missing_df["missing_count"] > 0
    ]

    if columns_with_missing.empty:
        st.success(
            "No missing values were detected."
        )

    else:
        figure = px.bar(
            columns_with_missing,
            x="column",
            y="missing_percentage",
            text="missing_percentage",
            title="Missing Values by Column",
            labels={
                "missing_percentage":
                    "Missing percentage (%)"
            },
        )

        figure.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )


# =========================================================
# DUPLICATES
# =========================================================

with tabs[2]:
    st.subheader("Duplicate-row analysis")

    duplicate_mask = df.duplicated(
        keep=False
    )

    duplicate_rows = df.loc[
        duplicate_mask
    ]

    duplicate_count = int(
        df.duplicated().sum()
    )

    duplicate_percentage = (
        duplicate_count / len(df) * 100
        if len(df)
        else 0
    )

    duplicate_columns = st.columns(2)

    duplicate_columns[0].metric(
        "Duplicate rows",
        f"{duplicate_count:,}",
    )

    duplicate_columns[1].metric(
        "Duplicate percentage",
        f"{duplicate_percentage:.2f}%",
    )

    if duplicate_rows.empty:
        st.success(
            "No duplicate rows were detected."
        )

    else:
        st.dataframe(
            duplicate_rows,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# OUTLIERS
# =========================================================

with tabs[3]:
    st.subheader("IQR outlier analysis")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    outlier_results = []

    for column in numeric_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        first_quartile = series.quantile(0.25)
        third_quartile = series.quantile(0.75)

        interquartile_range = (
            third_quartile
            - first_quartile
        )

        lower_limit = (
            first_quartile
            - 1.5 * interquartile_range
        )

        upper_limit = (
            third_quartile
            + 1.5 * interquartile_range
        )

        outlier_count = int(
            (
                (series < lower_limit)
                | (series > upper_limit)
            ).sum()
        )

        outlier_results.append({
            "column": column,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "outlier_count": outlier_count,
            "outlier_percentage": (
                outlier_count
                / len(series)
                * 100
            ),
        })

    outlier_df = pd.DataFrame(
        outlier_results
    )

    if outlier_df.empty:
        st.info(
            "No numerical columns are available."
        )

    else:
        outlier_df = outlier_df.sort_values(
            "outlier_percentage",
            ascending=False,
        )

        st.dataframe(
            outlier_df.style.format({
                "lower_limit": "{:.3f}",
                "upper_limit": "{:.3f}",
                "outlier_percentage": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        selected_numeric_column = st.selectbox(
            "View numerical distribution",
            options=numeric_columns,
        )

        figure = px.box(
            df,
            y=selected_numeric_column,
            points="outliers",
            title=(
                f"Outlier Distribution: "
                f"{selected_numeric_column}"
            ),
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )


# =========================================================
# CARDINALITY
# =========================================================

with tabs[4]:
    st.subheader("Categorical-cardinality analysis")

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    cardinality_results = []

    for column in categorical_columns:
        unique_count = df[column].nunique(
            dropna=True
        )

        unique_percentage = (
            unique_count / len(df) * 100
            if len(df)
            else 0
        )

        if unique_count <= 10:
            cardinality_level = "Low"
        elif unique_count <= 50:
            cardinality_level = "Medium"
        else:
            cardinality_level = "High"

        cardinality_results.append({
            "column": column,
            "unique_values": unique_count,
            "unique_percentage":
                unique_percentage,
            "cardinality_level":
                cardinality_level,
        })

    cardinality_df = pd.DataFrame(
        cardinality_results
    )

    if cardinality_df.empty:
        st.info(
            "No categorical columns were detected."
        )

    else:
        st.dataframe(
            cardinality_df.style.format({
                "unique_percentage": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# CLEAN DATASET
# =========================================================

with tabs[5]:
    st.subheader("Cleaning configuration")

    option_columns = st.columns(3)

    with option_columns[0]:
        remove_duplicates = st.checkbox(
            "Remove duplicate rows",
            value=True,
        )

    with option_columns[1]:
        missing_threshold_percentage = (
            st.slider(
                "Drop columns missing above",
                min_value=50,
                max_value=100,
                value=95,
                step=5,
                format="%d%%",
            )
        )

    with option_columns[2]:
        clip_outliers = st.checkbox(
            "Clip numerical IQR outliers",
            value=False,
        )

    st.caption(
        "Numerical missing values are filled using "
        "the median. Categorical missing values use "
        "the most frequent value."
    )

    if st.button(
        "Clean dataset",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            "Cleaning the dataset..."
        ):
            cleaned_df = clean(
                df,
                drop_duplicates=remove_duplicates,
                missing_threshold=(
                    missing_threshold_percentage
                    / 100
                ),
                clip_outliers=clip_outliers,
            )

            st.session_state.clean_df = (
                cleaned_df
            )

        st.success(
            "Dataset cleaned successfully."
        )

    if (
        "clean_df" in st.session_state
        and st.session_state.clean_df
        is not None
    ):
        cleaned_df = (
            st.session_state.clean_df
        )

        st.subheader("Cleaning result")

        comparison_columns = st.columns(4)

        comparison_columns[0].metric(
            "Original rows",
            f"{len(df):,}",
        )

        comparison_columns[1].metric(
            "Cleaned rows",
            f"{len(cleaned_df):,}",
            delta=(
                len(cleaned_df)
                - len(df)
            ),
        )

        comparison_columns[2].metric(
            "Original missing",
            f"{df.isna().sum().sum():,}",
        )

        comparison_columns[3].metric(
            "Cleaned missing",
            f"{cleaned_df.isna().sum().sum():,}",
        )

        st.dataframe(
            cleaned_df.head(100),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        parquet_buffer = BytesIO()

        cleaned_df.to_parquet(
            parquet_buffer,
            index=False,
        )

        download_columns = st.columns(2)

        with download_columns[0]:
            st.download_button(
                "Download cleaned CSV",
                data=csv_bytes,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with download_columns[1]:
            st.download_button(
                "Download cleaned Parquet",
                data=parquet_buffer.getvalue(),
                file_name="cleaned_dataset.parquet",
                mime="application/octet-stream",
                use_container_width=True,
            )