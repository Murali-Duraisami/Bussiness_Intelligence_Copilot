from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy import stats


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.core import statistical_summary


st.set_page_config(
    page_title="Statistical EDA | BI Copilot",
    page_icon="📈",
    layout="wide",
)


# =========================================================
# STYLE
# =========================================================

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

        .explanation-box {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            margin-bottom: 1rem;
        }

        .warning-box {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="page-header">
        <div class="page-title">
            Statistical Analysis and EDA
        </div>
        <div class="page-description">
            Explore distributions, correlations, descriptive
            statistics and hypothesis tests.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATASET
# =========================================================

if (
    "raw_df" not in st.session_state
    or st.session_state.raw_df is None
):
    st.warning(
        "No active dataset is available. Return to Home "
        "and upload a dataset or load demonstration data."
    )

    st.stop()


if (
    "clean_df" in st.session_state
    and st.session_state.clean_df is not None
):
    df = st.session_state.clean_df.copy()
    dataset_type = "Cleaned dataset"

else:
    df = st.session_state.raw_df.copy()
    dataset_type = "Raw dataset"


numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()


# =========================================================
# DATASET SUMMARY
# =========================================================

summary_columns = st.columns(5)

summary_columns[0].metric(
    "Dataset",
    dataset_type,
)

summary_columns[1].metric(
    "Rows",
    f"{len(df):,}",
)

summary_columns[2].metric(
    "Numerical columns",
    len(numeric_columns),
)

summary_columns[3].metric(
    "Categorical columns",
    len(categorical_columns),
)

summary_columns[4].metric(
    "Missing values",
    f"{df.isna().sum().sum():,}",
)


tabs = st.tabs(
    [
        "Descriptive statistics",
        "Distributions",
        "Correlations",
        "Categorical analysis",
        "Hypothesis testing",
        "Automated insights",
    ]
)


# =========================================================
# DESCRIPTIVE STATISTICS
# =========================================================

with tabs[0]:
    st.subheader("Descriptive statistics")

    st.markdown(
        """
        <div class="explanation-box">
            Descriptive statistics summarize central tendency,
            spread and distribution shape. Skewness measures
            asymmetry, while kurtosis measures tail heaviness.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not numeric_columns:
        st.info(
            "No numerical columns are available."
        )

    else:
        descriptive_df = statistical_summary(
            df[numeric_columns]
        )

        st.dataframe(
            descriptive_df.style.format(
                precision=4
            ),
            use_container_width=True,
        )

        selected_statistical_column = (
            st.selectbox(
                "Inspect numerical column",
                options=numeric_columns,
                key="descriptive_column",
            )
        )

        selected_series = pd.to_numeric(
            df[selected_statistical_column],
            errors="coerce",
        ).dropna()

        if len(selected_series):
            first_quartile = (
                selected_series.quantile(0.25)
            )

            third_quartile = (
                selected_series.quantile(0.75)
            )

            metric_columns = st.columns(6)

            metric_columns[0].metric(
                "Mean",
                f"{selected_series.mean():,.3f}",
            )

            metric_columns[1].metric(
                "Median",
                f"{selected_series.median():,.3f}",
            )

            metric_columns[2].metric(
                "Std. deviation",
                f"{selected_series.std():,.3f}",
            )

            metric_columns[3].metric(
                "IQR",
                (
                    f"{third_quartile - first_quartile:,.3f}"
                ),
            )

            metric_columns[4].metric(
                "Skewness",
                f"{selected_series.skew():.3f}",
            )

            metric_columns[5].metric(
                "Kurtosis",
                f"{selected_series.kurtosis():.3f}",
            )


# =========================================================
# DISTRIBUTIONS
# =========================================================

with tabs[1]:
    st.subheader("Distribution analysis")

    if not numeric_columns:
        st.info(
            "No numerical columns are available."
        )

    else:
        control_columns = st.columns(3)

        with control_columns[0]:
            selected_distribution_column = (
                st.selectbox(
                    "Numerical column",
                    options=numeric_columns,
                    key="distribution_column",
                )
            )

        with control_columns[1]:
            chart_type = st.selectbox(
                "Chart type",
                options=[
                    "Histogram",
                    "Box plot",
                    "Violin plot",
                ],
            )

        with control_columns[2]:
            number_of_bins = st.slider(
                "Histogram bins",
                min_value=10,
                max_value=100,
                value=30,
                disabled=(
                    chart_type != "Histogram"
                ),
            )

        if chart_type == "Histogram":
            figure = px.histogram(
                df,
                x=selected_distribution_column,
                nbins=number_of_bins,
                marginal="box",
                title=(
                    f"Distribution of "
                    f"{selected_distribution_column}"
                ),
            )

        elif chart_type == "Box plot":
            figure = px.box(
                df,
                y=selected_distribution_column,
                points="outliers",
                title=(
                    f"Box Plot of "
                    f"{selected_distribution_column}"
                ),
            )

        else:
            figure = px.violin(
                df,
                y=selected_distribution_column,
                box=True,
                points="outliers",
                title=(
                    f"Violin Plot of "
                    f"{selected_distribution_column}"
                ),
            )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        distribution_series = pd.to_numeric(
            df[selected_distribution_column],
            errors="coerce",
        ).dropna()

        st.subheader("Normality test")

        if len(distribution_series) >= 8:
            normality_statistic, normality_p = (
                stats.normaltest(
                    distribution_series
                )
            )

            normality_columns = st.columns(3)

            normality_columns[0].metric(
                "Test",
                "D'Agostino-Pearson",
            )

            normality_columns[1].metric(
                "Statistic",
                f"{normality_statistic:.4f}",
            )

            normality_columns[2].metric(
                "P-value",
                f"{normality_p:.6f}",
            )

            if normality_p > 0.05:
                st.success(
                    "There is insufficient evidence to "
                    "reject normality at the 5% level."
                )

            else:
                st.warning(
                    "The distribution significantly differs "
                    "from normal at the 5% level."
                )

        else:
            st.info(
                "At least eight valid observations are "
                "required for this normality test."
            )


# =========================================================
# CORRELATIONS
# =========================================================

with tabs[2]:
    st.subheader("Correlation analysis")

    if len(numeric_columns) < 2:
        st.info(
            "At least two numerical columns are required."
        )

    else:
        correlation_method = st.radio(
            "Correlation method",
            options=[
                "Pearson",
                "Spearman",
            ],
            horizontal=True,
        )

        method_name = (
            "pearson"
            if correlation_method == "Pearson"
            else "spearman"
        )

        correlation_matrix = (
            df[numeric_columns]
            .corr(method=method_name)
        )

        figure = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title=(
                f"{correlation_method} "
                f"Correlation Matrix"
            ),
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        st.dataframe(
            correlation_matrix.style.format(
                precision=3
            ),
            use_container_width=True,
        )

        upper_triangle = correlation_matrix.where(
            np.triu(
                np.ones(
                    correlation_matrix.shape
                ),
                k=1,
            ).astype(bool)
        )

        correlation_pairs = (
            upper_triangle.stack()
            .reset_index()
        )

        correlation_pairs.columns = [
            "first_variable",
            "second_variable",
            "correlation",
        ]

        correlation_pairs[
            "absolute_correlation"
        ] = correlation_pairs[
            "correlation"
        ].abs()

        correlation_pairs = (
            correlation_pairs.sort_values(
                "absolute_correlation",
                ascending=False,
            )
        )

        st.subheader(
            "Strongest numerical relationships"
        )

        st.dataframe(
            correlation_pairs.head(20).style.format({
                "correlation": "{:.4f}",
                "absolute_correlation": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        scatter_columns = st.columns(2)

        with scatter_columns[0]:
            scatter_x = st.selectbox(
                "X-axis",
                options=numeric_columns,
                key="scatter_x",
            )

        with scatter_columns[1]:
            scatter_y = st.selectbox(
                "Y-axis",
                options=numeric_columns,
                index=min(
                    1,
                    len(numeric_columns) - 1,
                ),
                key="scatter_y",
            )

        scatter_figure = px.scatter(
            df,
            x=scatter_x,
            y=scatter_y,
            trendline="ols",
            opacity=0.7,
            title=(
                f"{scatter_x} versus {scatter_y}"
            ),
        )

        st.plotly_chart(
            scatter_figure,
            use_container_width=True,
        )


# =========================================================
# CATEGORICAL ANALYSIS
# =========================================================

with tabs[3]:
    st.subheader("Categorical-variable analysis")

    available_categorical_columns = (
        categorical_columns.copy()
    )

    for column in numeric_columns:
        unique_count = df[column].nunique(
            dropna=True
        )

        if unique_count <= 20:
            available_categorical_columns.append(
                column
            )

    available_categorical_columns = list(
        dict.fromkeys(
            available_categorical_columns
        )
    )

    if not available_categorical_columns:
        st.info(
            "No categorical or low-cardinality "
            "columns were detected."
        )

    else:
        selected_category = st.selectbox(
            "Categorical column",
            options=available_categorical_columns,
        )

        categorical_summary = (
            df[selected_category]
            .value_counts(dropna=False)
            .rename_axis(selected_category)
            .reset_index(name="count")
        )

        categorical_summary["percentage"] = (
            categorical_summary["count"]
            / len(df)
            * 100
        )

        st.dataframe(
            categorical_summary.style.format({
                "percentage": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        category_figure = px.bar(
            categorical_summary,
            x=selected_category,
            y="count",
            color=selected_category,
            text="percentage",
            title=(
                f"Distribution of "
                f"{selected_category}"
            ),
        )

        category_figure.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        category_figure.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            category_figure,
            use_container_width=True,
        )


# =========================================================
# HYPOTHESIS TESTING
# =========================================================

with tabs[4]:
    st.subheader("Hypothesis testing")

    st.markdown(
        """
        <div class="explanation-box">
            A p-value below 0.05 indicates evidence against
            the null hypothesis. Statistical significance
            does not automatically mean practical or causal
            importance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    test_type = st.selectbox(
        "Select statistical test",
        options=[
            "Numeric versus binary category",
            "Numeric versus multiple categories",
            "Categorical versus categorical",
            "Numeric versus numeric",
        ],
    )

    if test_type == "Numeric versus binary category":
        binary_candidates = [
            column
            for column in df.columns
            if df[column].nunique(
                dropna=True
            ) == 2
        ]

        if not numeric_columns or not binary_candidates:
            st.info(
                "A numerical feature and a binary "
                "category are required."
            )

        else:
            selection_columns = st.columns(2)

            with selection_columns[0]:
                test_numeric = st.selectbox(
                    "Numerical feature",
                    options=numeric_columns,
                    key="binary_numeric",
                )

            with selection_columns[1]:
                binary_target = st.selectbox(
                    "Binary category",
                    options=binary_candidates,
                    key="binary_target",
                )

            if st.button(
                "Run Mann-Whitney U test",
                use_container_width=True,
            ):
                binary_values = (
                    df[binary_target]
                    .dropna()
                    .unique()
                )

                group_one = pd.to_numeric(
                    df.loc[
                        df[binary_target]
                        == binary_values[0],
                        test_numeric,
                    ],
                    errors="coerce",
                ).dropna()

                group_two = pd.to_numeric(
                    df.loc[
                        df[binary_target]
                        == binary_values[1],
                        test_numeric,
                    ],
                    errors="coerce",
                ).dropna()

                statistic, p_value = (
                    stats.mannwhitneyu(
                        group_one,
                        group_two,
                        alternative="two-sided",
                    )
                )

                test_result_columns = st.columns(4)

                test_result_columns[0].metric(
                    f"Median: {binary_values[0]}",
                    f"{group_one.median():,.3f}",
                )

                test_result_columns[1].metric(
                    f"Median: {binary_values[1]}",
                    f"{group_two.median():,.3f}",
                )

                test_result_columns[2].metric(
                    "U statistic",
                    f"{statistic:.3f}",
                )

                test_result_columns[3].metric(
                    "P-value",
                    f"{p_value:.6f}",
                )

                if p_value < 0.05:
                    st.success(
                        "A statistically significant "
                        "difference was detected."
                    )
                else:
                    st.info(
                        "No statistically significant "
                        "difference was detected."
                    )

                test_figure = px.box(
                    df,
                    x=df[binary_target].astype(str),
                    y=test_numeric,
                    color=df[
                        binary_target
                    ].astype(str),
                    title=(
                        f"{test_numeric} by "
                        f"{binary_target}"
                    ),
                    labels={
                        "x": binary_target,
                        "color": binary_target,
                    },
                )

                test_figure.update_layout(
                    showlegend=False
                )

                st.plotly_chart(
                    test_figure,
                    use_container_width=True,
                )

    elif test_type == "Numeric versus multiple categories":
        multiple_category_candidates = [
            column
            for column in df.columns
            if 3 <= df[column].nunique(
                dropna=True
            ) <= 20
        ]

        if (
            not numeric_columns
            or not multiple_category_candidates
        ):
            st.info(
                "A numerical feature and a category "
                "with at least three groups are required."
            )

        else:
            selection_columns = st.columns(2)

            with selection_columns[0]:
                test_numeric = st.selectbox(
                    "Numerical feature",
                    options=numeric_columns,
                    key="multi_numeric",
                )

            with selection_columns[1]:
                category_target = st.selectbox(
                    "Category",
                    options=multiple_category_candidates,
                    key="multi_category",
                )

            if st.button(
                "Run Kruskal-Wallis test",
                use_container_width=True,
            ):
                grouped_values = []

                for category_value in (
                    df[category_target]
                    .dropna()
                    .unique()
                ):
                    group = pd.to_numeric(
                        df.loc[
                            df[category_target]
                            == category_value,
                            test_numeric,
                        ],
                        errors="coerce",
                    ).dropna()

                    if len(group):
                        grouped_values.append(group)

                statistic, p_value = (
                    stats.kruskal(
                        *grouped_values
                    )
                )

                result_columns = st.columns(2)

                result_columns[0].metric(
                    "Kruskal statistic",
                    f"{statistic:.4f}",
                )

                result_columns[1].metric(
                    "P-value",
                    f"{p_value:.6f}",
                )

                if p_value < 0.05:
                    st.success(
                        "At least one category has a "
                        "significantly different distribution."
                    )
                else:
                    st.info(
                        "No significant distribution "
                        "difference was detected."
                    )

                figure = px.box(
                    df,
                    x=category_target,
                    y=test_numeric,
                    color=category_target,
                    title=(
                        f"{test_numeric} by "
                        f"{category_target}"
                    ),
                )

                figure.update_layout(
                    showlegend=False
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )

    elif test_type == "Categorical versus categorical":
        category_candidates = list(
            dict.fromkeys(
                available_categorical_columns
            )
        )

        if len(category_candidates) < 2:
            st.info(
                "At least two categorical variables "
                "are required."
            )

        else:
            selection_columns = st.columns(2)

            with selection_columns[0]:
                first_category = st.selectbox(
                    "First category",
                    options=category_candidates,
                    key="chi_first",
                )

            with selection_columns[1]:
                second_category = st.selectbox(
                    "Second category",
                    options=[
                        column
                        for column
                        in category_candidates
                        if column != first_category
                    ],
                    key="chi_second",
                )

            if st.button(
                "Run Chi-square test",
                use_container_width=True,
            ):
                contingency_table = pd.crosstab(
                    df[first_category],
                    df[second_category],
                )

                (
                    chi_square,
                    p_value,
                    degrees_of_freedom,
                    expected,
                ) = stats.chi2_contingency(
                    contingency_table
                )

                sample_size = (
                    contingency_table
                    .to_numpy()
                    .sum()
                )

                minimum_dimension = (
                    min(
                        contingency_table.shape
                    )
                    - 1
                )

                cramers_v = (
                    np.sqrt(
                        chi_square
                        / (
                            sample_size
                            * minimum_dimension
                        )
                    )
                    if minimum_dimension > 0
                    else np.nan
                )

                result_columns = st.columns(4)

                result_columns[0].metric(
                    "Chi-square",
                    f"{chi_square:.4f}",
                )

                result_columns[1].metric(
                    "P-value",
                    f"{p_value:.6f}",
                )

                result_columns[2].metric(
                    "Degrees of freedom",
                    degrees_of_freedom,
                )

                result_columns[3].metric(
                    "Cramér's V",
                    f"{cramers_v:.4f}",
                )

                st.dataframe(
                    contingency_table,
                    use_container_width=True,
                )

                if p_value < 0.05:
                    st.success(
                        "A statistically significant "
                        "association was detected."
                    )
                else:
                    st.info(
                        "No statistically significant "
                        "association was detected."
                    )

    else:
        if len(numeric_columns) < 2:
            st.info(
                "At least two numerical columns "
                "are required."
            )

        else:
            selection_columns = st.columns(2)

            with selection_columns[0]:
                first_numeric = st.selectbox(
                    "First numerical variable",
                    options=numeric_columns,
                    key="correlation_first",
                )

            with selection_columns[1]:
                second_numeric = st.selectbox(
                    "Second numerical variable",
                    options=[
                        column
                        for column in numeric_columns
                        if column != first_numeric
                    ],
                    key="correlation_second",
                )

            if st.button(
                "Run correlation tests",
                use_container_width=True,
            ):
                valid_rows = df[
                    [
                        first_numeric,
                        second_numeric,
                    ]
                ].dropna()

                pearson_value, pearson_p = (
                    stats.pearsonr(
                        valid_rows[first_numeric],
                        valid_rows[second_numeric],
                    )
                )

                spearman_value, spearman_p = (
                    stats.spearmanr(
                        valid_rows[first_numeric],
                        valid_rows[second_numeric],
                    )
                )

                correlation_results = pd.DataFrame([
                    {
                        "test": "Pearson",
                        "correlation":
                            pearson_value,
                        "p_value": pearson_p,
                    },
                    {
                        "test": "Spearman",
                        "correlation":
                            spearman_value,
                        "p_value": spearman_p,
                    },
                ])

                st.dataframe(
                    correlation_results.style.format({
                        "correlation": "{:.4f}",
                        "p_value": "{:.6f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )


# =========================================================
# AUTOMATED INSIGHTS
# =========================================================

with tabs[5]:
    st.subheader("Automated statistical insights")

    insights = []

    if numeric_columns:
        skewness = (
            df[numeric_columns]
            .skew()
            .sort_values(
                key=abs,
                ascending=False,
            )
        )

        most_skewed_column = skewness.index[0]

        insights.append(
            f"{most_skewed_column} has the strongest "
            f"distribution skewness at "
            f"{skewness.iloc[0]:.3f}."
        )

    if len(numeric_columns) >= 2:
        correlation_matrix = (
            df[numeric_columns]
            .corr(method="spearman")
        )

        upper_triangle = correlation_matrix.where(
            np.triu(
                np.ones(
                    correlation_matrix.shape
                ),
                k=1,
            ).astype(bool)
        )

        strongest_correlation = (
            upper_triangle.abs()
            .stack()
            .sort_values(
                ascending=False
            )
        )

        if len(strongest_correlation):
            first_column, second_column = (
                strongest_correlation.index[0]
            )

            actual_correlation = (
                correlation_matrix.loc[
                    first_column,
                    second_column,
                ]
            )

            insights.append(
                f"The strongest Spearman relationship is "
                f"between {first_column} and "
                f"{second_column} "
                f"({actual_correlation:.3f})."
            )

    for column in categorical_columns:
        most_common_value = (
            df[column]
            .mode(dropna=True)
        )

        if len(most_common_value):
            insights.append(
                f"The most common value in {column} "
                f"is {most_common_value.iloc[0]}."
            )

    if not insights:
        st.info(
            "Not enough information is available "
            "for automated insights."
        )

    else:
        for number, insight in enumerate(
            insights,
            start=1,
        ):
            st.markdown(
                f"**{number}.** {insight}"
            )

    st.markdown(
        """
        <div class="warning-box">
            Statistical relationships do not prove causality.
            Interpret all findings using business and domain
            knowledge.
        </div>
        """,
        unsafe_allow_html=True,
    )