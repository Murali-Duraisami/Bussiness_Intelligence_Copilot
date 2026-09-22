# Intelligent Business Intelligence Copilot

Notebook-first, open-source BI and machine-learning project. It loads common tabular files or SQL data, profiles and cleans them, performs statistical analysis and EDA, trains supervised/unsupervised/ANN models, explains predictions, exports PDF/PPTX reports, and provides a professional Streamlit UI.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name intelligent-bi-copilot --display-name "Python (Intelligent BI Copilot)"
jupyter lab
```

Run notebooks in numeric order. To launch the app:

```powershell
streamlit run app/Home.py
```

The app supports CSV, TSV/TXT, Excel, JSON/JSONL, Parquet, Feather, Pickle, XML, SQLite, SAS, SPSS and Stata. SQLAlchemy URLs support other databases when the relevant free driver is installed.

## Important

Models are decision-support tools, not guarantees. Validate assumptions, leakage, fairness, and domain suitability before production use. Never upload untrusted pickle files.
