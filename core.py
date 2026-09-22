import numpy as np
import pandas as pd
from scipy import stats

def profile(df):
    rows=len(df)
    out=pd.DataFrame({'column':df.columns,'dtype':df.dtypes.astype(str).values,'non_null':df.notna().sum().values,'missing':df.isna().sum().values,'missing_pct':(df.isna().mean()*100).round(2).values,'unique':[df[c].nunique(dropna=True) for c in df]})
    return {'rows':rows,'columns':df.shape[1],'duplicates':int(df.duplicated().sum()),'memory_mb':round(df.memory_usage(deep=True).sum()/1048576,3),'numeric':len(df.select_dtypes(include=np.number).columns),'categorical':len(df.select_dtypes(exclude=np.number).columns)},out

def clean(df, drop_duplicates=True, missing_threshold=0.95, clip_outliers=False):
    x=df.copy(); x.columns=[str(c).strip().replace(' ','_') for c in x]
    x=x.loc[:,x.isna().mean()<missing_threshold]
    if drop_duplicates: x=x.drop_duplicates()
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]): x[c]=x[c].fillna(x[c].median())
        else:
            mode=x[c].mode(dropna=True); x[c]=x[c].fillna(mode.iloc[0] if len(mode) else 'Missing')
    if clip_outliers:
        for c in x.select_dtypes(include=np.number):
            q1,q3=x[c].quantile([.25,.75]); iqr=q3-q1
            if iqr>0: x[c]=x[c].clip(q1-1.5*iqr,q3+1.5*iqr)
    return x

def statistical_summary(df):
    numeric=df.select_dtypes(include=np.number)
    result=numeric.describe().T
    result['variance']=numeric.var(); result['skewness']=numeric.skew(); result['kurtosis']=numeric.kurt()
    result['normality_p_value']=[stats.normaltest(numeric[c].dropna()).pvalue if numeric[c].notna().sum()>=8 else np.nan for c in numeric]
    return result

def quality_score(df):
    completeness=1-df.isna().mean().mean(); uniqueness=1-min(df.duplicated().mean(),1)
    valid_names=1-df.columns.duplicated().mean(); return round(100*(.6*completeness+.3*uniqueness+.1*valid_names),1)
