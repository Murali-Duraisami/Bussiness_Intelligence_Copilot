from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_data


ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"
EXPLAIN_DIR = ARTIFACT_DIR / "explainability"
UNSUPERVISED_DIR = ARTIFACT_DIR / "unsupervised"

DEMO_DATA_FILE = (
    ROOT
    / "data"
    / "uploads"
    / "sample_business_data.csv"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Intelligent BI Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background-color: #0f172a;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .hero-container {
            padding: 2.2rem 2.4rem;
            border-radius: 22px;
            background:
                linear-gradient(
                    120deg,
                    #172554 0%,
                    #1d4ed8 58%,
                    #0ea5e9 100%
                );
            color: white;
            margin-bottom: 1.5rem;
            box-shadow:
                0 16px 40px rgba(30, 64, 175, 0.22);
        }

        .hero-title {
            font-size: 2.35rem;
            font-weight: 750;
            margin-bottom: 0.5rem;
            line-height: 1.15;
        }

        .hero-subtitle {
            font-size: 1.04rem;
            line-height: 1.7;
            color: #dbeafe;
            max-width: 900px;
        }

        .status-badge {
            display: inline-block;
            padding: 0.34rem 0.72rem;
            margin-right: 0.4rem;
            margin-top: 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.25);
            font-size: 0.82rem;
            font-weight: 600;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #172554;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
        }

        .workflow-card {
            min-height: 180px;
            padding: 1.2rem;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }

        .workflow-number {
            width: 34px;
            height: 34px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            background: #dbeafe;
            color: #1d4ed8;
            font-weight: 750;
            margin-bottom: 0.8rem;
        }

        .workflow-heading {
            color: #172554;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .workflow-description {
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 1rem;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            overflow: hidden;
        }

        .success-box {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
        }

        .information-box {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
        }

        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
            color: #64748b;
            font-size: 0.82rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def read_json_file(path):
    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception:
        return {}


def initialize_session_state():
    default_values = {
        "raw_df": None,
        "clean_df": None,
        "dataset_name": None,
        "target_column": None,
        "problem_type": None,
        "trained_models": {},
        "active_model": None,
        "upload_version": 0,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_dataset_outputs():
    keys_to_reset = [
        "clean_df",
        "target_column",
        "problem_type",
        "trained_models",
        "active_model",
    ]

    for key in keys_to_reset:
        st.session_state[key] = (
            {} if key == "trained_models" else None
        )


def set_active_dataset(
    dataframe,
    dataset_name,
):
    reset_dataset_outputs()

    st.session_state.raw_df = dataframe
    st.session_state.dataset_name = (
        dataset_name
    )

    st.session_state.upload_version += 1


def load_uploaded_dataset(uploaded_file):
    return load_data(uploaded_file)


def artifact_exists(relative_path):
    return (ROOT / relative_path).exists()


initialize_session_state()


# =========================================================
# LOAD COMPLETED NOTEBOOK RESULTS
# =========================================================

calibrated_metrics = read_json_file(
    MODEL_DIR / "calibrated_model_metrics.json"
)

final_model_summary = read_json_file(
    MODEL_DIR / "final_selection_summary.json"
)

unsupervised_summary = read_json_file(
    UNSUPERVISED_DIR / "unsupervised_summary.json"
)

explanation_summary = read_json_file(
    EXPLAIN_DIR / "explanation_summary.json"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 📊 BI Copilot")

    st.caption(
        "Automated analytics and machine learning"
    )

    st.divider()

    st.markdown("### Data source")

    data_source = st.radio(
        "Choose a source",
        options=[
            "Upload dataset",
            "Use demonstration data",
        ],
        label_visibility="collapsed",
    )

    if data_source == "Upload dataset":
        uploaded_file = st.file_uploader(
            "Upload a tabular dataset",
            type=[
                "csv",
                "tsv",
                "txt",
                "xlsx",
                "xls",
                "xlsm",
                "json",
                "jsonl",
                "parquet",
                "pq",
                "feather",
                "xml",
                "html",
                "htm",
                "dta",
                "sav",
                "sas7bdat",
                "xpt",
            ],
        )

        if uploaded_file is not None:
            file_identifier = (
                uploaded_file.name,
                uploaded_file.size,
            )

            if (
                st.session_state.get(
                    "loaded_file_identifier"
                )
                != file_identifier
            ):
                try:
                    dataframe = (
                        load_uploaded_dataset(
                            uploaded_file
                        )
                    )

                    set_active_dataset(
                        dataframe=dataframe,
                        dataset_name=uploaded_file.name,
                    )

                    st.session_state[
                        "loaded_file_identifier"
                    ] = file_identifier

                    st.success(
                        "Dataset loaded successfully."
                    )

                except Exception as error:
                    st.error(
                        f"Unable to load dataset: "
                        f"{error}"
                    )

    else:
        if st.button(
            "Load demonstration dataset",
            use_container_width=True,
            type="primary",
        ):
            try:
                demonstration_dataframe = (
                    load_data(DEMO_DATA_FILE)
                )

                set_active_dataset(
                    dataframe=demonstration_dataframe,
                    dataset_name=(
                        DEMO_DATA_FILE.name
                    ),
                )

                st.success(
                    "Demonstration dataset loaded."
                )

            except Exception as error:
                st.error(
                    f"Unable to load demonstration "
                    f"dataset: {error}"
                )

    st.divider()

    st.markdown("### Notebook status")

    notebook_status = [
        (
            "Data ingestion",
            artifact_exists(
                "artifacts/cleaned_data.parquet"
            ),
        ),
        (
            "Model training",
            artifact_exists(
                "artifacts/models/"
                "final_selected_model.joblib"
            ),
        ),
        (
            "Calibration",
            artifact_exists(
                "artifacts/models/"
                "final_calibrated_model.joblib"
            ),
        ),
        (
            "Explainability",
            artifact_exists(
                "artifacts/explainability/"
                "explanation_summary.json"
            ),
        ),
        (
            "PDF report",
            artifact_exists(
                "artifacts/reports/"
                "Intelligent_BI_Copilot_Report.pdf"
            ),
        ),
    ]

    for status_name, completed in notebook_status:
        icon = "✅" if completed else "○"

        st.write(f"{icon} {status_name}")

    st.divider()

    st.caption(
        "All processing remains local. "
        "No paid API is required."
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">
            Intelligent Business Intelligence Copilot
        </div>
        <div class="hero-subtitle">
            Convert raw business data into statistical analysis,
            interactive visualizations, machine-learning models,
            explainable predictions and professional reports.
        </div>
        <span class="status-badge">Data Quality</span>
        <span class="status-badge">Statistical EDA</span>
        <span class="status-badge">Supervised ML</span>
        <span class="status-badge">Unsupervised ML</span>
        <span class="status-badge">ANN</span>
        <span class="status-badge">Explainable AI</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# COMPLETED MODEL METRICS
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Validated demonstration results'
    '</div>',
    unsafe_allow_html=True,
)

metric_columns = st.columns(5)

metric_columns[0].metric(
    "Selected model",
    final_model_summary.get(
        "final_model",
        "Not available",
    ),
)

metric_columns[1].metric(
    "Accuracy",
    (
        f"{calibrated_metrics.get('accuracy', 0) * 100:.1f}%"
        if calibrated_metrics
        else "N/A"
    ),
)

metric_columns[2].metric(
    "Macro F1",
    (
        f"{calibrated_metrics.get('macro_f1', 0):.3f}"
        if calibrated_metrics
        else "N/A"
    ),
)

metric_columns[3].metric(
    "ROC-AUC",
    (
        f"{calibrated_metrics.get('roc_auc', 0):.3f}"
        if calibrated_metrics
        else "N/A"
    ),
)

metric_columns[4].metric(
    "Customer segments",
    unsupervised_summary.get(
        "selected_k",
        "N/A",
    ),
)


# =========================================================
# ACTIVE DATASET
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Active dataset'
    '</div>',
    unsafe_allow_html=True,
)

if st.session_state.raw_df is None:
    st.markdown(
        """
        <div class="information-box">
            Select a dataset from the sidebar. You can upload
            your own file or load the included demonstration
            dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    active_df = st.session_state.raw_df

    dataset_columns = st.columns(4)

    dataset_columns[0].metric(
        "Dataset",
        st.session_state.dataset_name,
    )

    dataset_columns[1].metric(
        "Rows",
        f"{len(active_df):,}",
    )

    dataset_columns[2].metric(
        "Columns",
        f"{active_df.shape[1]:,}",
    )

    dataset_columns[3].metric(
        "Missing values",
        f"{active_df.isna().sum().sum():,}",
    )

    st.markdown("#### Dataset preview")

    st.dataframe(
        active_df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = active_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download current dataset",
        data=csv_data,
        file_name="current_dataset.csv",
        mime="text/csv",
    )


# =========================================================
# WORKFLOW
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Copilot workflow'
    '</div>',
    unsafe_allow_html=True,
)

workflow_items = [
    (
        "01",
        "Load data",
        (
            "Upload CSV, Excel, JSON, Parquet and "
            "other tabular formats."
        ),
    ),
    (
        "02",
        "Profile and clean",
        (
            "Detect missing values, duplicates, "
            "outliers and quality issues."
        ),
    ),
    (
        "03",
        "Explore",
        (
            "Generate statistical tests, distributions, "
            "correlations and visual insights."
        ),
    ),
    (
        "04",
        "Build models",
        (
            "Choose a target and compare supervised, "
            "unsupervised and ANN models."
        ),
    ),
    (
        "05",
        "Explain",
        (
            "Use SHAP, permutation importance and "
            "individual prediction explanations."
        ),
    ),
    (
        "06",
        "Deliver",
        (
            "Ask natural-language questions and export "
            "PDF and PowerPoint reports."
        ),
    ),
]

first_row = st.columns(3)
second_row = st.columns(3)

for index, item in enumerate(workflow_items):
    number, heading, description = item

    target_column = (
        first_row[index]
        if index < 3
        else second_row[index - 3]
    )

    with target_column:
        st.markdown(
            f"""
            <div class="workflow-card">
                <div class="workflow-number">
                    {number}
                </div>
                <div class="workflow-heading">
                    {heading}
                </div>
                <div class="workflow-description">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Intelligent BI Copilot · Notebook-first analytics ·
        Free and open-source Python tools
    </div>
    """,
    unsafe_allow_html=True,
)