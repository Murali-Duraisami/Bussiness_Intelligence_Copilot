from pathlib import Path
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text

SUPPORTED={'.csv','.tsv','.txt','.xlsx','.xls','.xlsm','.json','.jsonl','.parquet','.pq','.feather','.pkl','.pickle','.xml','.html','.htm','.db','.sqlite','.sqlite3','.sas7bdat','.xpt','.sav','.dta'}

def extension(file):
    name=str(file) if isinstance(file,(str,Path)) else getattr(file,'name','')
    ext=Path(name).suffix.lower()
    if not ext: raise ValueError('Cannot determine file type.')
    return ext

def _rewind(file):
    if hasattr(file,'seek'): file.seek(0)

def sqlite_tables(path):
    with sqlite3.connect(path) as c:
        q="SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        return pd.read_sql_query(q,c)['name'].tolist()

def load_sql(url, query):
    if not query.strip().lower().startswith(('select','with')): raise ValueError('Only SELECT/WITH queries are allowed.')
    engine=create_engine(url)
    with engine.connect() as c: return pd.read_sql_query(text(query),c)

def load_data(file, sheet_name=0, table_name=None, query=None, **kwargs):
    ext=extension(file); _rewind(file)
    if ext not in SUPPORTED: raise ValueError(f'Unsupported {ext}. Supported: {sorted(SUPPORTED)}')
    if ext in {'.csv','.tsv','.txt'}:
        sep='\t' if ext=='.tsv' else (None if ext=='.txt' else ',')
        last=None
        for enc in ['utf-8','utf-8-sig','cp1252','latin-1']:
            try: _rewind(file); df=pd.read_csv(file,sep=sep,engine='python' if sep is None else 'c',encoding=enc,**kwargs); break
            except UnicodeDecodeError as e: last=e
        else: raise last
    elif ext in {'.xlsx','.xls','.xlsm'}: df=pd.read_excel(file,sheet_name=sheet_name,**kwargs)
    elif ext in {'.json','.jsonl'}:
        try: df=pd.read_json(file,lines=ext=='.jsonl',**kwargs)
        except ValueError: _rewind(file); df=pd.read_json(file,lines=ext!='.jsonl',**kwargs)
    elif ext in {'.parquet','.pq'}: df=pd.read_parquet(file,**kwargs)
    elif ext=='.feather': df=pd.read_feather(file,**kwargs)
    elif ext in {'.pkl','.pickle'}: df=pd.read_pickle(file,**kwargs)
    elif ext=='.xml': df=pd.read_xml(file,**kwargs)
    elif ext in {'.html','.htm'}: df=pd.read_html(file,**kwargs)[0]
    elif ext in {'.sas7bdat','.xpt'}: df=pd.read_sas(file,**kwargs)
    elif ext=='.sav': df=pd.read_spss(file,**kwargs)
    elif ext=='.dta': df=pd.read_stata(file,**kwargs)
    else:
        if not isinstance(file,(str,Path)): raise ValueError('Save SQLite upload locally before reading.')
        tables=sqlite_tables(file); chosen=table_name or (tables[0] if tables else None)
        if not chosen: raise ValueError('No tables found.')
        if chosen not in tables: raise ValueError(f'Unknown table. Available: {tables}')
        with sqlite3.connect(file) as c: df=pd.read_sql_query(f'SELECT * FROM "{chosen.replace(chr(34),chr(34)*2)}"',c)
    if not isinstance(df,pd.DataFrame): df=pd.DataFrame(df)
    df.columns=[str(c).strip() for c in df.columns]
    return df
