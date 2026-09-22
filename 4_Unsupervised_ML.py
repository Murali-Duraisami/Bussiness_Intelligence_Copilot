from __future__ import annotations

import io
import json
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Unsupervised ML | BI Copilot",
    page_icon="🧩",
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
        <h1>Unsupervised Machine Learning</h1>
        <p>
            Discover hidden groups, compare cluster quality, visualize
            observations using PCA, identify unusual records and create
            reusable clustering artifacts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------

def get_active_dataframe() -> pd.DataFrame | None:
    clean_df = st.session_state.get("clean_df")
    raw_df = st.session_state.get("raw_df")

    if isinstance(clean_df, pd.DataFrame) and not clean_df.empty:
        return clean_df.copy()

    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        return raw_df.copy()

    return None


def possible_identifier_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    identifiers: list[str] = []

    for column in dataframe.columns:
        column_name = column.lower().strip()

        name_looks_like_id = (
            column_name == "id"
            or column_name.endswith("_id")
            or column_name.startswith("id_")
            or "identifier" in column_name
        )

        unique_ratio = dataframe[column].nunique(
            dropna=True
        ) / max(len(dataframe), 1)

        if name_looks_like_id or unique_ratio >= 0.98:
            identifiers.append(column)

    return identifiers


def create_preprocessor(
    dataframe: pd.DataFrame,
) -> ColumnTransformer:
    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns
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


def safe_cluster_metrics(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    unique_labels = np.unique(labels)

    valid_labels = unique_labels[
        unique_labels != -1
    ]

    if len(valid_labels) < 2:
        return {
            "silhouette_score": np.nan,
            "davies_bouldin_score": np.nan,
            "calinski_harabasz_score": np.nan,
        }

    valid_mask = labels != -1
    valid_features = feature_matrix[valid_mask]
    valid_cluster_labels = labels[valid_mask]

    if len(valid_features) <= len(valid_labels):
        return {
            "silhouette_score": np.nan,
            "davies_bouldin_score": np.nan,
            "calinski_harabasz_score": np.nan,
        }

    silhouette_sample = min(
        5000,
        len(valid_features),
    )

    try:
        silhouette = silhouette_score(
            valid_features,
            valid_cluster_labels,
            sample_size=silhouette_sample,
            random_state=42,
        )
    except Exception:
        silhouette = np.nan

    try:
        davies_bouldin = davies_bouldin_score(
            valid_features,
            valid_cluster_labels,
        )
    except Exception:
        davies_bouldin = np.nan

    try:
        calinski_harabasz = calinski_harabasz_score(
            valid_features,
            valid_cluster_labels,
        )
    except Exception:
        calinski_harabasz = np.nan

    return {
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
        "calinski_harabasz_score": calinski_harabasz,
    }


def serialize_object(obj: Any) -> bytes:
    memory_file = io.BytesIO()
    joblib.dump(obj, memory_file)
    memory_file.seek(0)
    return memory_file.getvalue()


def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------
# Load dataset
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
# Configuration
# ---------------------------------------------------------

st.subheader("1. Configure unsupervised analysis")

all_columns = list(df.columns)
default_excluded_columns = possible_identifier_columns(df)

previous_target = st.session_state.get("target_column")

if (
    previous_target in df.columns
    and previous_target not in default_excluded_columns
):
    default_excluded_columns.append(previous_target)


excluded_columns = st.multiselect(
    "Columns to exclude",
    options=all_columns,
    default=default_excluded_columns,
    help=(
        "Exclude IDs, target columns and fields that should not influence "
        "statistical similarity."
    ),
)


available_features = [
    column
    for column in all_columns
    if column not in excluded_columns
]


selected_features = st.multiselect(
    "Features used for clustering",
    options=available_features,
    default=available_features,
)


if not selected_features:
    st.error("Select at least one feature for clustering.")
    st.stop()


reference_columns = st.multiselect(
    "Reference columns for cluster profiling",
    options=[
        column
        for column in all_columns
        if column not in selected_features
    ],
    default=[
        column
        for column in [previous_target]
        if column in df.columns
        and column not in selected_features
    ],
    help=(
        "Reference columns do not influence clustering. They are used only "
        "to understand the resulting groups."
    ),
)


configuration_one, configuration_two, configuration_three = st.columns(3)

with configuration_one:
    maximum_possible_clusters = max(
        2,
        min(15, len(df) - 1),
    )

    maximum_clusters = st.slider(
        "Maximum K-Means clusters",
        min_value=2,
        max_value=maximum_possible_clusters,
        value=min(10, maximum_possible_clusters),
    )

with configuration_two:
    random_state = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

with configuration_three:
    anomaly_percentage = st.slider(
        "Expected anomaly percentage",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )


st.write("**DBSCAN configuration**")

dbscan_one, dbscan_two = st.columns(2)

with dbscan_one:
    dbscan_eps = st.slider(
        "DBSCAN epsilon",
        min_value=0.1,
        max_value=5.0,
        value=0.8,
        step=0.1,
    )

with dbscan_two:
    dbscan_min_samples = st.slider(
        "DBSCAN minimum samples",
        min_value=2,
        max_value=50,
        value=5,
        step=1,
    )


st.markdown(
    """
    <div class="status-info">
        The system tests multiple K-Means solutions and selects the number
        of clusters with the highest silhouette score. A silhouette score
        near 1 indicates separated clusters, while a value near 0 indicates
        overlapping groups.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Run analysis
# ---------------------------------------------------------

run_button = st.button(
    "Run unsupervised analysis",
    type="primary",
    width="stretch",
)


if run_button:
    analysis_df = df[selected_features].copy()

    if len(analysis_df) < 10:
        st.error(
            "At least 10 records are required for clustering."
        )
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    try:
        status.info("Preprocessing numeric and categorical features...")

        preprocessor = create_preprocessor(analysis_df)
        processed_features = preprocessor.fit_transform(
            analysis_df
        )

        processed_features = np.asarray(
            processed_features,
            dtype=float,
        )

        feature_names = preprocessor.get_feature_names_out().tolist()

        progress.progress(10)

        kmeans_rows: list[dict[str, float]] = []
        kmeans_models: dict[int, KMeans] = {}

        cluster_range = range(
            2,
            maximum_clusters + 1,
        )

        for position, cluster_count in enumerate(cluster_range):
            status.info(
                f"Evaluating K-Means with {cluster_count} clusters..."
            )

            kmeans = KMeans(
                n_clusters=cluster_count,
                n_init=20,
                max_iter=500,
                random_state=int(random_state),
            )

            labels = kmeans.fit_predict(
                processed_features
            )

            metrics = safe_cluster_metrics(
                processed_features,
                labels,
            )

            kmeans_rows.append(
                {
                    "number_of_clusters": cluster_count,
                    "inertia": kmeans.inertia_,
                    **metrics,
                }
            )

            kmeans_models[cluster_count] = kmeans

            progress.progress(
                int(
                    10
                    + 40
                    * (
                        (position + 1)
                        / len(list(cluster_range))
                    )
                )
            )

        kmeans_evaluation = pd.DataFrame(
            kmeans_rows
        )

        valid_silhouette_rows = kmeans_evaluation.dropna(
            subset=["silhouette_score"]
        )

        if valid_silhouette_rows.empty:
            selected_cluster_count = 2
        else:
            selected_cluster_count = int(
                valid_silhouette_rows.sort_values(
                    "silhouette_score",
                    ascending=False,
                ).iloc[0]["number_of_clusters"]
            )

        selected_kmeans = kmeans_models[
            selected_cluster_count
        ]

        kmeans_labels = selected_kmeans.labels_

        selected_kmeans_metrics = safe_cluster_metrics(
            processed_features,
            kmeans_labels,
        )

        status.info("Creating PCA visualization...")

        pca_component_count = min(
            2,
            processed_features.shape[0],
            processed_features.shape[1],
        )

        pca = PCA(
            n_components=pca_component_count,
            random_state=int(random_state),
        )

        pca_values = pca.fit_transform(
            processed_features
        )

        if pca_component_count == 1:
            pca_values = np.column_stack(
                [
                    pca_values[:, 0],
                    np.zeros(len(pca_values)),
                ]
            )
            explained_variance = [
                float(pca.explained_variance_ratio_[0]),
                0.0,
            ]
        else:
            explained_variance = (
                pca.explained_variance_ratio_.tolist()
            )

        progress.progress(60)

        algorithm_rows = [
            {
                "algorithm": "K-Means",
                "clusters": selected_cluster_count,
                **selected_kmeans_metrics,
            }
        ]

        hierarchical_model = None
        hierarchical_labels = None

        if len(processed_features) <= 5000:
            status.info(
                "Evaluating hierarchical clustering..."
            )

            hierarchical_model = AgglomerativeClustering(
                n_clusters=selected_cluster_count,
                linkage="ward",
            )

            hierarchical_labels = (
                hierarchical_model.fit_predict(
                    processed_features
                )
            )

            hierarchical_metrics = safe_cluster_metrics(
                processed_features,
                hierarchical_labels,
            )

            algorithm_rows.append(
                {
                    "algorithm": "Hierarchical",
                    "clusters": selected_cluster_count,
                    **hierarchical_metrics,
                }
            )

        progress.progress(70)

        status.info("Evaluating DBSCAN clustering...")

        dbscan_model = DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
        )

        dbscan_labels = dbscan_model.fit_predict(
            processed_features
        )

        dbscan_cluster_count = len(
            set(dbscan_labels)
        ) - (
            1 if -1 in dbscan_labels else 0
        )

        dbscan_noise_count = int(
            np.sum(dbscan_labels == -1)
        )

        dbscan_metrics = safe_cluster_metrics(
            processed_features,
            dbscan_labels,
        )

        algorithm_rows.append(
            {
                "algorithm": "DBSCAN",
                "clusters": dbscan_cluster_count,
                **dbscan_metrics,
            }
        )

        progress.progress(80)

        status.info("Detecting potential anomalies...")

        isolation_forest = IsolationForest(
            n_estimators=250,
            contamination=anomaly_percentage / 100,
            random_state=int(random_state),
            n_jobs=-1,
        )

        isolation_labels = isolation_forest.fit_predict(
            processed_features
        )

        anomaly_scores = isolation_forest.decision_function(
            processed_features
        )

        anomaly_labels = np.where(
            isolation_labels == -1,
            "Anomaly",
            "Normal",
        )

        progress.progress(90)

        clustered_data = df.copy()
        clustered_data["kmeans_cluster"] = kmeans_labels
        clustered_data["dbscan_cluster"] = dbscan_labels
        clustered_data["anomaly_label"] = anomaly_labels
        clustered_data["anomaly_score"] = anomaly_scores
        clustered_data["pca_component_1"] = pca_values[:, 0]
        clustered_data["pca_component_2"] = pca_values[:, 1]

        if hierarchical_labels is not None:
            clustered_data[
                "hierarchical_cluster"
            ] = hierarchical_labels

        numeric_profile_columns = (
            df.select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        profile_numeric_columns = list(
            dict.fromkeys(
                [
                    column
                    for column in (
                        selected_features
                        + reference_columns
                    )
                    if column in numeric_profile_columns
                ]
            )
        )

        cluster_sizes = (
            clustered_data["kmeans_cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("kmeans_cluster")
            .reset_index(name="records")
        )

        cluster_sizes["percentage"] = (
            cluster_sizes["records"]
            / len(clustered_data)
            * 100
        )

        if profile_numeric_columns:
            numeric_profiles = (
                clustered_data.groupby(
                    "kmeans_cluster"
                )[profile_numeric_columns]
                .mean(numeric_only=True)
                .reset_index()
            )

            cluster_profiles = cluster_sizes.merge(
                numeric_profiles,
                on="kmeans_cluster",
                how="left",
            )
        else:
            cluster_profiles = cluster_sizes.copy()

        algorithm_comparison = pd.DataFrame(
            algorithm_rows
        )

        anomaly_data = clustered_data.loc[
            clustered_data["anomaly_label"]
            == "Anomaly"
        ].copy()

        model_bundle = {
            "preprocessor": preprocessor,
            "kmeans_model": selected_kmeans,
            "pca_model": pca,
            "isolation_forest_model": isolation_forest,
            "dbscan_model": dbscan_model,
            "hierarchical_model": hierarchical_model,
            "selected_features": selected_features,
            "feature_names": feature_names,
            "selected_cluster_count": selected_cluster_count,
        }

        st.session_state.unsupervised_preprocessor = preprocessor
        st.session_state.kmeans_model = selected_kmeans
        st.session_state.pca_model = pca
        st.session_state.isolation_forest_model = isolation_forest
        st.session_state.dbscan_model = dbscan_model
        st.session_state.hierarchical_model = hierarchical_model
        st.session_state.unsupervised_model_bundle = model_bundle
        st.session_state.clustered_data = clustered_data
        st.session_state.cluster_profiles = cluster_profiles
        st.session_state.kmeans_evaluation = kmeans_evaluation
        st.session_state.algorithm_comparison = algorithm_comparison
        st.session_state.anomaly_data = anomaly_data
        st.session_state.unsupervised_features = selected_features
        st.session_state.unsupervised_feature_names = feature_names
        st.session_state.selected_cluster_count = (
            selected_cluster_count
        )
        st.session_state.pca_explained_variance = (
            explained_variance
        )
        st.session_state.dbscan_noise_count = (
            dbscan_noise_count
        )
        st.session_state.dbscan_cluster_count = (
            dbscan_cluster_count
        )

        progress.progress(100)
        status.empty()
        progress.empty()

        st.success(
            "Unsupervised analysis completed successfully."
        )

    except Exception as error:
        status.empty()
        progress.empty()

        st.error(
            f"Unsupervised analysis failed: {error}"
        )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

clustered_data = st.session_state.get(
    "clustered_data"
)
cluster_profiles = st.session_state.get(
    "cluster_profiles"
)
kmeans_evaluation = st.session_state.get(
    "kmeans_evaluation"
)
algorithm_comparison = st.session_state.get(
    "algorithm_comparison"
)
anomaly_data = st.session_state.get(
    "anomaly_data"
)


if (
    isinstance(clustered_data, pd.DataFrame)
    and not clustered_data.empty
    and isinstance(kmeans_evaluation, pd.DataFrame)
):
    selected_cluster_count = (
        st.session_state.get(
            "selected_cluster_count",
            0,
        )
    )

    selected_row = kmeans_evaluation.loc[
        kmeans_evaluation[
            "number_of_clusters"
        ] == selected_cluster_count
    ]

    if not selected_row.empty:
        selected_silhouette = selected_row.iloc[0][
            "silhouette_score"
        ]
    else:
        selected_silhouette = np.nan

    anomaly_count = len(anomaly_data)
    anomaly_rate = (
        anomaly_count / len(clustered_data) * 100
    )

    pca_variance = st.session_state.get(
        "pca_explained_variance",
        [0.0, 0.0],
    )

    total_pca_variance = sum(pca_variance)

    st.divider()
    st.subheader("2. Unsupervised analysis summary")

    metric_one, metric_two, metric_three, metric_four = st.columns(4)

    metric_one.metric(
        "Selected clusters",
        selected_cluster_count,
    )

    metric_two.metric(
        "Silhouette score",
        (
            f"{selected_silhouette:.4f}"
            if pd.notna(selected_silhouette)
            else "Not available"
        ),
    )

    metric_three.metric(
        "Potential anomalies",
        f"{anomaly_count:,}",
        delta=f"{anomaly_rate:.2f}% of records",
        delta_color="off",
    )

    metric_four.metric(
        "PCA variance shown",
        f"{total_pca_variance:.2%}",
    )

    if (
        pd.notna(selected_silhouette)
        and selected_silhouette < 0.25
    ):
        st.warning(
            "The silhouette score is low. The identified groups overlap "
            "and should be validated before assigning business segment "
            "names."
        )
    elif (
        pd.notna(selected_silhouette)
        and selected_silhouette < 0.50
    ):
        st.info(
            "The clustering structure is moderate. Review cluster profiles "
            "before using the segments operationally."
        )
    elif pd.notna(selected_silhouette):
        st.success(
            "The clustering result shows reasonably separated groups."
        )

    results_tab_one, results_tab_two, results_tab_three, results_tab_four = (
        st.tabs(
            [
                "K-Means selection",
                "Cluster visualization",
                "Cluster profiles",
                "Anomalies",
            ]
        )
    )

    with results_tab_one:
        evaluation_display = kmeans_evaluation.copy()

        numeric_columns = evaluation_display.select_dtypes(
            include=np.number
        ).columns

        evaluation_display[numeric_columns] = (
            evaluation_display[numeric_columns]
            .round(4)
        )

        st.dataframe(
            evaluation_display,
            width="stretch",
            hide_index=True,
        )

        evaluation_chart = px.line(
            kmeans_evaluation,
            x="number_of_clusters",
            y="silhouette_score",
            markers=True,
            title="Silhouette score by number of clusters",
            labels={
                "number_of_clusters": "Number of clusters",
                "silhouette_score": "Silhouette score",
            },
        )

        evaluation_chart.add_vline(
            x=selected_cluster_count,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="Selected",
        )

        evaluation_chart.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            evaluation_chart,
            width="stretch",
        )

        st.write("**Algorithm comparison**")

        comparison_display = algorithm_comparison.copy()

        comparison_numeric = (
            comparison_display.select_dtypes(
                include=np.number
            ).columns
        )

        comparison_display[comparison_numeric] = (
            comparison_display[
                comparison_numeric
            ].round(4)
        )

        st.dataframe(
            comparison_display,
            width="stretch",
            hide_index=True,
        )

        dbscan_clusters = st.session_state.get(
            "dbscan_cluster_count",
            0,
        )
        dbscan_noise = st.session_state.get(
            "dbscan_noise_count",
            0,
        )

        st.caption(
            f"DBSCAN created {dbscan_clusters} clusters and marked "
            f"{dbscan_noise:,} records as noise."
        )

    with results_tab_two:
        visualization_df = clustered_data.copy()

        visualization_df["Cluster"] = (
            visualization_df["kmeans_cluster"]
            .astype(str)
        )

        pca_chart = px.scatter(
            visualization_df,
            x="pca_component_1",
            y="pca_component_2",
            color="Cluster",
            hover_data=[
                column
                for column in df.columns[:8]
                if column in visualization_df.columns
            ],
            title="K-Means clusters projected into two PCA dimensions",
            labels={
                "pca_component_1": (
                    f"PC1 ({pca_variance[0]:.1%} variance)"
                ),
                "pca_component_2": (
                    f"PC2 ({pca_variance[1]:.1%} variance)"
                ),
            },
        )

        pca_chart.update_layout(
            template="plotly_white",
            legend_title_text="Cluster",
        )

        st.plotly_chart(
            pca_chart,
            width="stretch",
        )

        st.caption(
            "PCA compresses the transformed features into two dimensions "
            "for visualization. Some information is not visible in this "
            "two-dimensional chart."
        )

    with results_tab_three:
        st.dataframe(
            cluster_profiles.round(4),
            width="stretch",
            hide_index=True,
        )

        size_chart = px.bar(
            cluster_profiles,
            x="kmeans_cluster",
            y="records",
            color="kmeans_cluster",
            text=cluster_profiles["percentage"].map(
                lambda value: f"{value:.1f}%"
            ),
            title="Records in each cluster",
            labels={
                "kmeans_cluster": "Cluster",
                "records": "Number of records",
            },
        )

        size_chart.update_layout(
            template="plotly_white",
            showlegend=False,
        )

        st.plotly_chart(
            size_chart,
            width="stretch",
        )

        categorical_profile_options = [
            column
            for column in reference_columns
            if column in clustered_data.columns
            and not pd.api.types.is_numeric_dtype(
                clustered_data[column]
            )
        ]

        if categorical_profile_options:
            selected_profile_column = st.selectbox(
                "Categorical cluster profile",
                options=categorical_profile_options,
            )

            categorical_profile = pd.crosstab(
                clustered_data["kmeans_cluster"],
                clustered_data[selected_profile_column],
                normalize="index",
            ) * 100

            st.dataframe(
                categorical_profile.round(2),
                width="stretch",
            )

    with results_tab_four:
        anomaly_chart = px.scatter(
            clustered_data,
            x="pca_component_1",
            y="pca_component_2",
            color="anomaly_label",
            color_discrete_map={
                "Normal": "#2563eb",
                "Anomaly": "#dc2626",
            },
            title="Isolation Forest anomaly detection",
            labels={
                "pca_component_1": "PCA component 1",
                "pca_component_2": "PCA component 2",
                "anomaly_label": "Record type",
            },
        )

        anomaly_chart.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            anomaly_chart,
            width="stretch",
        )

        st.write("**Potential anomaly records**")

        st.dataframe(
            anomaly_data.sort_values(
                "anomaly_score",
                ascending=True,
            ),
            width="stretch",
            hide_index=True,
        )

        st.warning(
            "Anomalies should be reviewed before removal. They may be "
            "data errors, rare events or valuable business cases."
        )

    st.subheader("3. Download unsupervised outputs")

    download_one, download_two, download_three = st.columns(3)

    with download_one:
        st.download_button(
            "Download model bundle",
            data=serialize_object(
                st.session_state[
                    "unsupervised_model_bundle"
                ]
            ),
            file_name="unsupervised_model_bundle.joblib",
            mime="application/octet-stream",
            width="stretch",
        )

    with download_two:
        st.download_button(
            "Download clustered dataset",
            data=to_csv_bytes(clustered_data),
            file_name="clustered_data.csv",
            mime="text/csv",
            width="stretch",
        )

    with download_three:
        st.download_button(
            "Download cluster profiles",
            data=to_csv_bytes(cluster_profiles),
            file_name="cluster_profiles.csv",
            mime="text/csv",
            width="stretch",
        )

    download_four, download_five, download_six = st.columns(3)

    with download_four:
        st.download_button(
            "Download K-Means evaluation",
            data=to_csv_bytes(kmeans_evaluation),
            file_name="kmeans_evaluation.csv",
            mime="text/csv",
            width="stretch",
        )

    with download_five:
        st.download_button(
            "Download anomalies",
            data=to_csv_bytes(anomaly_data),
            file_name="potential_anomalies.csv",
            mime="text/csv",
            width="stretch",
        )

    summary = {
        "selected_features": st.session_state.get(
            "unsupervised_features",
            [],
        ),
        "processed_feature_count": len(
            st.session_state.get(
                "unsupervised_feature_names",
                [],
            )
        ),
        "selected_cluster_count": selected_cluster_count,
        "silhouette_score": (
            float(selected_silhouette)
            if pd.notna(selected_silhouette)
            else None
        ),
        "pca_component_1_variance": float(
            pca_variance[0]
        ),
        "pca_component_2_variance": float(
            pca_variance[1]
        ),
        "total_pca_variance": float(
            total_pca_variance
        ),
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_rate,
        "dbscan_clusters": st.session_state.get(
            "dbscan_cluster_count",
            0,
        ),
        "dbscan_noise_records": st.session_state.get(
            "dbscan_noise_count",
            0,
        ),
        "interpretation_note": (
            "Cluster labels represent statistical similarity and do not "
            "automatically establish causal business segments."
        ),
    }

    with download_six:
        st.download_button(
            "Download analysis summary",
            data=json.dumps(
                summary,
                indent=2,
                default=str,
            ),
            file_name="unsupervised_summary.json",
            mime="application/json",
            width="stretch",
        )

else:
    st.info(
        "Select the clustering features and click "
        "**Run unsupervised analysis**."
    )