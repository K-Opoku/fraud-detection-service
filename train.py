#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier



def clean_data():
    df= pd.read_csv('/workspaces/Project_1_ML_Zoomcamp/Data/Synthetic_Financial_datasets_log.csv')
    del df['isFlaggedFraud']
    df.columns= df.columns.str.lower().str.replace(' ','_')
    strings=['type','nameorig','namedest' ]
    for i in strings:
        df[i]= df[i].str.lower().str.replace(' ','_')

    # dropping identifiers columns
    df= df.drop(columns=['nameorig','namedest'])
    df.columns
    # Data Cleaning
    df.duplicated().sum()
    df.isnull().sum()



    # dropping missing values rows in the target
    df= df.dropna(subset=['isfraud'])



    # Applied log transform to mitigate outliers
    outlier_col=['amount', 'oldbalanceorg', 'newbalanceorig', 'oldbalancedest', 'newbalancedest']
    for i in outlier_col:
        df[i]=np.log1p(df[i])


    return df


categorical=['type']
numerical=['step','amount','oldbalanceorg','newbalanceorig','oldbalancedest','newbalancedest']



def train_model(df):
    # ## Validation Framework

    df_fulltrain,df_test=train_test_split(df, test_size=0.2, random_state=42)


    # ## Retraining the final model on the fulltrain dataset
    y_fulltrain=df_fulltrain.isfraud.values
    del df_fulltrain['isfraud']


    fulltrain_dict= df_fulltrain[categorical+numerical].to_dict(orient='records')
    pipeline=make_pipeline(DictVectorizer(),XGBClassifier(n_estimators=90,random_state=42,max_depth=5,min_child_weight=1,gamma=0,scale_pos_weight=10,colsample_bytree=0.7,eta=0.1))
    pipeline.fit(fulltrain_dict,y_fulltrain)
    return pipeline





def save_model(pipeline,filename='final_model_pipeline.pkl'):
    joblib.dump(pipeline, filename)
    print(f'Model saved to {filename}')


df= clean_data()
pipeline= train_model(df)
save_model(pipeline)

