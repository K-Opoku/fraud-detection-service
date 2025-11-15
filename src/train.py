#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



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



   

    return df


categorical=['type']
numerical=['step','amount','oldbalanceorg','newbalanceorig','oldbalancedest','newbalancedest']
outlier_col=['amount', 'oldbalanceorg', 'newbalanceorig', 'oldbalancedest', 'newbalancedest']



def train_model(df):
    # ## Validation Framework

    df_fulltrain,df_test=train_test_split(df, test_size=0.2, random_state=42)


    # ## Retraining the final model on the fulltrain dataset
    y_fulltrain=df_fulltrain.isfraud.values
    del df_fulltrain['isfraud']


   
    

    passthrough_feature = ['step']

    log_transformer= Pipeline(steps=[('log1p',FunctionTransformer(np.log1p,validate=False) )])
    categorical_transformer=Pipeline(steps=[('onehotencode',OneHotEncoder(handle_unknown='ignore'))])
    preprocessor=ColumnTransformer(transformers=[('log_transform',log_transformer,outlier_col),('categorical_transform',categorical_transformer,categorical),('passthrough_trans','passthrough',passthrough_feature)])
    final_pipeline=Pipeline(steps=[('preprocessor',preprocessor),('xgb_model',XGBClassifier(n_estimators=90,random_state=42,max_depth=5,min_child_weight=1,gamma=0,scale_pos_weight=10,colsample_bytree=0.7,eta=0.1))])


    final_pipeline.fit(df_fulltrain,y_fulltrain)
    return  final_pipeline





def save_model(final_pipeline,filename='final_model_pipeline.pkl'):
    joblib.dump(final_pipeline, filename)
    print(f'Model saved to {filename}')


df= clean_data()
pipeline= train_model(df)
save_model(pipeline)

