import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,IsolationForest
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def infer_problem(y):
    n=y.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(y) or n<=max(20,int(len(y)*.05)): return 'classification'
    return 'regression'

def split_xy(df,target):
    x=df.drop(columns=[target]); y=df[target]
    keep=y.notna(); return x.loc[keep],y.loc[keep]

def pipeline_for(x,model):
    num=x.select_dtypes(include=np.number).columns.tolist(); cat=[c for c in x.columns if c not in num]
    prep=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    return Pipeline([('preprocessor',prep),('model',model)])

def models(problem):
    if problem=='classification': out={'Logistic Regression':LogisticRegression(max_iter=2000),'Random Forest':RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1,class_weight='balanced'),'Extra Trees':ExtraTreesClassifier(n_estimators=200,random_state=42,n_jobs=-1,class_weight='balanced'),'Gradient Boosting':GradientBoostingClassifier(random_state=42),'KNN':KNeighborsClassifier(),'ANN (MLP)':MLPClassifier(hidden_layer_sizes=(128,64),early_stopping=True,max_iter=300,random_state=42)}
    else: out={'Linear Regression':LinearRegression(),'Ridge':Ridge(),'Random Forest':RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1),'Extra Trees':ExtraTreesRegressor(n_estimators=200,random_state=42,n_jobs=-1),'Gradient Boosting':GradientBoostingRegressor(random_state=42),'KNN':KNeighborsRegressor(),'ANN (MLP)':MLPRegressor(hidden_layer_sizes=(128,64),early_stopping=True,max_iter=300,random_state=42)}
    try:
        from xgboost import XGBClassifier,XGBRegressor
        out['XGBoost']=XGBClassifier(n_estimators=250,random_state=42,n_jobs=-1) if problem=='classification' else XGBRegressor(n_estimators=250,random_state=42,n_jobs=-1)
    except ImportError: pass
    try:
        from lightgbm import LGBMClassifier,LGBMRegressor
        out['LightGBM']=LGBMClassifier(n_estimators=250,random_state=42,verbosity=-1) if problem=='classification' else LGBMRegressor(n_estimators=250,random_state=42,verbosity=-1)
    except ImportError: pass
    try:
        from catboost import CatBoostClassifier,CatBoostRegressor
        out['CatBoost']=CatBoostClassifier(iterations=250,verbose=False,random_seed=42) if problem=='classification' else CatBoostRegressor(iterations=250,verbose=False,random_seed=42)
    except ImportError: pass
    return out

def train_compare(df,target,problem='auto',selected=None,test_size=.2):
    x,y=split_xy(df,target); problem=infer_problem(y) if problem=='auto' else problem
    strat=y if problem=='classification' and y.value_counts().min()>=2 else None
    xt,xv,yt,yv=train_test_split(x,y,test_size=test_size,random_state=42,stratify=strat)
    catalog=models(problem); names=selected or list(catalog); rows=[]; fitted={}
    for name in names:
        try:
            p=pipeline_for(x,catalog[name]); p.fit(xt,yt); pred=p.predict(xv); fitted[name]=p
            if problem=='classification': rows.append({'model':name,'accuracy':accuracy_score(yv,pred),'f1_weighted':f1_score(yv,pred,average='weighted',zero_division=0)})
            else: rows.append({'model':name,'mae':mean_absolute_error(yv,pred),'rmse':mean_squared_error(yv,pred)**.5,'r2':r2_score(yv,pred)})
        except Exception as e: rows.append({'model':name,'error':str(e)})
    result=pd.DataFrame(rows)
    if problem=='classification' and 'f1_weighted' in result: result=result.sort_values('f1_weighted',ascending=False)
    elif 'r2' in result: result=result.sort_values('r2',ascending=False)
    return problem,result,fitted,(xv,yv)

def cluster(df,algorithm='KMeans',n_clusters=3):
    x=df.select_dtypes(include=np.number).replace([np.inf,-np.inf],np.nan); x=SimpleImputer(strategy='median').fit_transform(x); x=StandardScaler().fit_transform(x)
    if algorithm=='KMeans': model=KMeans(n_clusters=n_clusters,random_state=42,n_init=10)
    elif algorithm=='Hierarchical': model=AgglomerativeClustering(n_clusters=n_clusters)
    else: model=DBSCAN(eps=.5,min_samples=5)
    labels=model.fit_predict(x); score=silhouette_score(x,labels) if len(set(labels))>1 and len(set(labels))<len(labels) else np.nan
    coords=PCA(n_components=2,random_state=42).fit_transform(x); return labels,score,pd.DataFrame({'PC1':coords[:,0],'PC2':coords[:,1],'cluster':labels.astype(str)})
