from __future__ import annotations

import os
import sqlite3
import tempfile
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="SQL Database | BI Copilot",
    page_icon="🗄️",
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
        <h1>SQL Database Connection</h1>
        <p>
            Import data from SQLite, PostgreSQL, MySQL, Microsoft SQL Server
            or another SQLAlchemy-compatible database and use it throughout
            the complete BI Copilot workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session-state functions
# ---------------------------------------------------------

def clear_previous_analysis() -> None:
    analysis_keys = [
        "clean_df",
        "target_column",
        "problem_type",
        "feature_columns",
        "trained_models",
        "active_model",
        "active_model_name",
        "model_results",
        "model_predictions",
        "all_model_predictions",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "model_failures",
        "cv_folds",
        "unsupervised_preprocessor",
        "kmeans_model",
        "pca_model",
        "isolation_forest_model",
        "dbscan_model",
        "hierarchical_model",
        "unsupervised_model_bundle",
        "clustered_data",
        "cluster_profiles",
        "kmeans_evaluation",
        "algorithm_comparison",
        "anomaly_data",
        "selected_cluster_count",
        "pca_explained_variance",
        "dbscan_noise_count",
        "dbscan_cluster_count",
        "builtin_feature_importance",
        "grouped_builtin_importance",
        "permutation_importance_df",
        "shap_values",
        "shap_processed_features",
        "shap_feature_names",
        "shap_sample_raw",
        "shap_importance_df",
        "grouped_shap_df",
        "latest_individual_prediction",
        "batch_prediction_results",
        "prediction_history",
        "generated_pdf_report",
        "generated_powerpoint_report",
        "generated_summary_json",
        "generated_analysis_package",
        "copilot_messages",
    ]

    for key in analysis_keys:
        st.session_state.pop(key, None)


def activate_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
    source_type: str,
) -> None:
    clear_previous_analysis()

    st.session_state.raw_df = dataframe.copy()
    st.session_state.clean_df = dataframe.copy()
    st.session_state.dataset_name = dataset_name
    st.session_state.data_source_type = source_type
    st.session_state.upload_version = (
        st.session_state.get(
            "upload_version",
            0,
        )
        + 1
    )


def is_read_only_query(query: str) -> bool:
    cleaned_query = query.strip()

    cleaned_query = re.sub(
        r"(--[^\n]*|/\*.*?\*/)",
        " ",
        cleaned_query,
        flags=re.DOTALL,
    ).strip()

    normalized_query = cleaned_query.lower()

    allowed_starts = (
        "select",
        "with",
        "pragma table_info",
    )

    if not normalized_query.startswith(
        allowed_starts
    ):
        return False

    dangerous_keywords = [
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " alter ",
        " truncate ",
        " create ",
        " replace ",
        " attach ",
        " detach ",
        " vacuum ",
        " reindex ",
    ]

    padded_query = f" {normalized_query} "

    return not any(
        keyword in padded_query
        for keyword in dangerous_keywords
    )


def display_dataframe_summary(
    dataframe: pd.DataFrame,
) -> None:
    metric_one, metric_two, metric_three, metric_four = st.columns(4)

    metric_one.metric(
        "Rows",
        f"{len(dataframe):,}",
    )

    metric_two.metric(
        "Columns",
        f"{dataframe.shape[1]:,}",
    )

    metric_three.metric(
        "Missing values",
        f"{int(dataframe.isna().sum().sum()):,}",
    )

    metric_four.metric(
        "Duplicate rows",
        f"{int(dataframe.duplicated().sum()):,}",
    )

    st.dataframe(
        dataframe.head(100),
        width="stretch",
        hide_index=True,
    )


# re is required by is_read_only_query
import re


# ---------------------------------------------------------
# SQL source selection
# ---------------------------------------------------------

source_tab_one, source_tab_two = st.tabs(
    [
        "Upload SQLite database",
        "Connect using SQLAlchemy URL",
    ]
)


# ---------------------------------------------------------
# SQLite file upload
# ---------------------------------------------------------

with source_tab_one:
    st.subheader("Upload a SQLite database")

    st.markdown(
        """
        <div class="status-info">
            Upload a <b>.db</b>, <b>.sqlite</b> or <b>.sqlite3</b> file.
            The application will list the available tables without modifying
            the original database.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sqlite_file = st.file_uploader(
        "SQLite database file",
        type=[
            "db",
            "sqlite",
            "sqlite3",
        ],
        key="sqlite_database_upload",
    )

    if sqlite_file is not None:
        temporary_path = None

        try:
            suffix = Path(
                sqlite_file.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temporary_file:
                temporary_file.write(
                    sqlite_file.getbuffer()
                )

                temporary_path = (
                    temporary_file.name
                )

            connection = sqlite3.connect(
                temporary_path
            )

            tables_df = pd.read_sql_query(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """,
                connection,
            )

            table_names = (
                tables_df["name"].tolist()
                if not tables_df.empty
                else []
            )

            if not table_names:
                st.warning(
                    "No user tables were found in this SQLite database."
                )

            else:
                st.success(
                    f"Found {len(table_names)} database table(s)."
                )

                sqlite_mode = st.radio(
                    "Import method",
                    options=[
                        "Select a table",
                        "Write a read-only SQL query",
                    ],
                    horizontal=True,
                    key="sqlite_import_mode",
                )

                maximum_rows = st.number_input(
                    "Maximum rows to import",
                    min_value=100,
                    max_value=5_000_000,
                    value=100_000,
                    step=1000,
                    key="sqlite_maximum_rows",
                )

                if sqlite_mode == "Select a table":
                    selected_table = st.selectbox(
                        "Database table",
                        options=table_names,
                        key="sqlite_selected_table",
                    )

                    quoted_table = (
                        selected_table.replace(
                            '"',
                            '""',
                        )
                    )

                    count_query = (
                        f'SELECT COUNT(*) AS row_count '
                        f'FROM "{quoted_table}"'
                    )

                    row_count = int(
                        pd.read_sql_query(
                            count_query,
                            connection,
                        ).iloc[0]["row_count"]
                    )

                    st.metric(
                        "Rows available in table",
                        f"{row_count:,}",
                    )

                    preview_query = (
                        f'SELECT * FROM "{quoted_table}" '
                        f"LIMIT 20"
                    )

                    preview_df = pd.read_sql_query(
                        preview_query,
                        connection,
                    )

                    st.write("**Table preview**")

                    st.dataframe(
                        preview_df,
                        width="stretch",
                        hide_index=True,
                    )

                    if st.button(
                        "Load selected SQLite table",
                        type="primary",
                        width="stretch",
                        key="load_sqlite_table",
                    ):
                        import_query = (
                            f'SELECT * FROM "{quoted_table}" '
                            f"LIMIT {int(maximum_rows)}"
                        )

                        imported_df = pd.read_sql_query(
                            import_query,
                            connection,
                        )

                        activate_dataframe(
                            imported_df,
                            (
                                f"{sqlite_file.name} — "
                                f"{selected_table}"
                            ),
                            "SQLite database",
                        )

                        st.session_state.sql_imported_df = (
                            imported_df
                        )

                        st.success(
                            f"Loaded {len(imported_df):,} rows from "
                            f"{selected_table}."
                        )

                else:
                    sqlite_query = st.text_area(
                        "Read-only SQL query",
                        value=(
                            f'SELECT * FROM "{table_names[0]}" '
                            "LIMIT 1000"
                        ),
                        height=180,
                        key="sqlite_custom_query",
                    )

                    if st.button(
                        "Run SQLite query",
                        type="primary",
                        width="stretch",
                        key="run_sqlite_query",
                    ):
                        if not is_read_only_query(
                            sqlite_query
                        ):
                            st.error(
                                "Only read-only SELECT or WITH queries "
                                "are allowed."
                            )

                        else:
                            query_df = pd.read_sql_query(
                                sqlite_query,
                                connection,
                            )

                            if len(query_df) > maximum_rows:
                                query_df = query_df.head(
                                    int(maximum_rows)
                                )

                                st.warning(
                                    f"The result was limited to "
                                    f"{int(maximum_rows):,} rows."
                                )

                            activate_dataframe(
                                query_df,
                                (
                                    f"{sqlite_file.name} — "
                                    "custom query"
                                ),
                                "SQLite query",
                            )

                            st.session_state.sql_imported_df = (
                                query_df
                            )

                            st.success(
                                f"Loaded {len(query_df):,} query rows."
                            )

            connection.close()

        except Exception as error:
            st.error(
                f"SQLite import failed: {error}"
            )

        finally:
            if (
                temporary_path
                and os.path.exists(temporary_path)
            ):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass


# ---------------------------------------------------------
# SQLAlchemy connection
# ---------------------------------------------------------

with source_tab_two:
    st.subheader("Connect to a SQL server")

    st.markdown(
        """
        <div class="status-info">
            Enter a SQLAlchemy connection URL. Credentials are used only
            for the current Streamlit session and are not written to the
            generated reports.
        </div>
        """,
        unsafe_allow_html=True,
    )

    database_type = st.selectbox(
        "Database type",
        options=[
            "PostgreSQL",
            "MySQL",
            "Microsoft SQL Server",
            "SQLite path",
            "Other SQLAlchemy database",
        ],
    )

    example_urls = {
        "PostgreSQL": (
            "postgresql+psycopg2://username:password@localhost:5432/database"
        ),
        "MySQL": (
            "mysql+pymysql://username:password@localhost:3306/database"
        ),
        "Microsoft SQL Server": (
            "mssql+pyodbc://username:password@server/database"
            "?driver=ODBC+Driver+17+for+SQL+Server"
        ),
        "SQLite path": (
            "sqlite:///E:/path/to/database.db"
        ),
        "Other SQLAlchemy database": (
            "dialect+driver://username:password@host/database"
        ),
    }

    connection_url = st.text_input(
        "SQLAlchemy connection URL",
        value="",
        placeholder=example_urls[database_type],
        type="password",
    )

    st.caption(
        f"Example: {example_urls[database_type]}"
    )

    connection_schema = st.text_input(
        "Schema name — optional",
        value="",
        help=(
            "For PostgreSQL, this is commonly public. Leave empty to use "
            "the database default."
        ),
    )

    if st.button(
        "Connect and inspect database",
        type="primary",
        width="stretch",
    ):
        if not connection_url.strip():
            st.error(
                "Enter a SQLAlchemy connection URL."
            )

        else:
            try:
                from sqlalchemy import create_engine, inspect

                engine = create_engine(
                    connection_url,
                    pool_pre_ping=True,
                )

                with engine.connect() as connection:
                    connection.exec_driver_sql(
                        "SELECT 1"
                    )

                inspector = inspect(engine)

                schema_value = (
                    connection_schema.strip()
                    or None
                )

                server_tables = inspector.get_table_names(
                    schema=schema_value
                )

                st.session_state.sqlalchemy_engine = (
                    engine
                )
                st.session_state.sql_server_tables = (
                    server_tables
                )
                st.session_state.sql_server_schema = (
                    schema_value
                )

                st.success(
                    f"Connection succeeded. Found "
                    f"{len(server_tables)} table(s)."
                )

            except ImportError:
                st.error(
                    "SQLAlchemy is not installed. Run: "
                    "pip install sqlalchemy"
                )

            except Exception as error:
                st.error(
                    f"Database connection failed: {error}"
                )


    server_engine = st.session_state.get(
        "sqlalchemy_engine"
    )

    server_tables = st.session_state.get(
        "sql_server_tables",
        [],
    )

    server_schema = st.session_state.get(
        "sql_server_schema"
    )

    if (
        server_engine is not None
        and server_tables
    ):
        server_import_mode = st.radio(
            "Server import method",
            options=[
                "Select a table",
                "Write a read-only SQL query",
            ],
            horizontal=True,
            key="server_import_mode",
        )

        server_maximum_rows = st.number_input(
            "Maximum server rows to import",
            min_value=100,
            max_value=5_000_000,
            value=100_000,
            step=1000,
            key="server_maximum_rows",
        )

        if server_import_mode == "Select a table":
            selected_server_table = st.selectbox(
                "Server table",
                options=server_tables,
            )

            if st.button(
                "Load server table",
                type="primary",
                width="stretch",
            ):
                try:
                    from sqlalchemy import (
                        MetaData,
                        Table,
                        select,
                    )

                    metadata = MetaData()

                    reflected_table = Table(
                        selected_server_table,
                        metadata,
                        schema=server_schema,
                        autoload_with=server_engine,
                    )

                    statement = select(
                        reflected_table
                    ).limit(
                        int(server_maximum_rows)
                    )

                    with server_engine.connect() as connection:
                        server_df = pd.read_sql(
                            statement,
                            connection,
                        )

                    activate_dataframe(
                        server_df,
                        selected_server_table,
                        f"{database_type} table",
                    )

                    st.session_state.sql_imported_df = (
                        server_df
                    )

                    st.success(
                        f"Loaded {len(server_df):,} rows from "
                        f"{selected_server_table}."
                    )

                except Exception as error:
                    st.error(
                        f"Table import failed: {error}"
                    )

        else:
            server_query = st.text_area(
                "Read-only SQL query",
                value="SELECT * FROM your_table",
                height=180,
                key="server_custom_query",
            )

            if st.button(
                "Run server query",
                type="primary",
                width="stretch",
            ):
                if not is_read_only_query(
                    server_query
                ):
                    st.error(
                        "Only read-only SELECT or WITH queries are allowed."
                    )

                else:
                    try:
                        from sqlalchemy import text

                        with server_engine.connect() as connection:
                            server_query_df = pd.read_sql_query(
                                text(server_query),
                                connection,
                            )

                        if len(
                            server_query_df
                        ) > server_maximum_rows:
                            server_query_df = (
                                server_query_df.head(
                                    int(
                                        server_maximum_rows
                                    )
                                )
                            )

                            st.warning(
                                f"The result was limited to "
                                f"{int(server_maximum_rows):,} rows."
                            )

                        activate_dataframe(
                            server_query_df,
                            "SQL custom query",
                            f"{database_type} query",
                        )

                        st.session_state.sql_imported_df = (
                            server_query_df
                        )

                        st.success(
                            f"Loaded {len(server_query_df):,} query rows."
                        )

                    except Exception as error:
                        st.error(
                            f"SQL query failed: {error}"
                        )


# ---------------------------------------------------------
# Imported data summary
# ---------------------------------------------------------

sql_imported_df = st.session_state.get(
    "sql_imported_df"
)

if (
    isinstance(sql_imported_df, pd.DataFrame)
    and not sql_imported_df.empty
):
    st.divider()
    st.subheader("Imported SQL dataset")

    display_dataframe_summary(
        sql_imported_df
    )

    st.success(
        "The SQL data is now the active dataset. Continue with "
        "**Data Quality**, **Statistical EDA**, **Supervised ML** or "
        "**Unsupervised ML**."
    )

    st.download_button(
        "Download imported SQL data as CSV",
        data=sql_imported_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="sql_imported_dataset.csv",
        mime="text/csv",
        width="stretch",
    )


st.warning(
    "Use database accounts with read-only permissions. Avoid connecting "
    "with an administrator or write-enabled production account."
)