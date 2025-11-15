import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Literal
import pandas as pd



pipeline=joblib.load('final_model_pipeline.pkl')

app=FastAPI(title='Fraud Detection Service')

def predict_single(transaction):
    df = pd.DataFrame([transaction])
    result=pipeline.predict_proba(df)[0,1]
    return result

class Transaction(BaseModel):
    step: int
    type: Literal['cash_in','cash_out','debit','payment','transfer']
    amount: float=Field(...,ge=0.0)
    oldbalanceorg: float=Field(...,ge=0.0)  
    newbalanceorig: float=Field(...,ge=0.0)
    oldbalancedest: float=Field(...,ge=0.0)
    newbalancedest: float=Field(...,ge=0.0)

class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud: bool

@app.post('/predict')
def predict(transaction:Transaction)->PredictionResponse:
    transaction_dict=[transaction.dict()]
    fraud_probability=predict_single(transaction.model_dump())
    fraud= fraud_probability>=0.038
    return PredictionResponse(fraud_probability=fraud_probability,fraud=fraud)

if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)
