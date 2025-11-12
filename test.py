from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,precision_score,recall_score,precision_recall_curve,average_precision_score
import joblib

# ## Testing the final (XGBoost) model

test_dict=df_test[categorical+numerical].to_dict(orient='records')
y_score_test=pipeline.predict_proba(test_dict)[:,1]
y_pred_test=y_score_test>=0.038
print('The average precision score of the final model:',average_precision_score(y_test,y_pred_test))
print('The recall score of the final model:',recall_score(y_test,y_pred_test))
print('The precision score of the final model:', precision_score(y_test,y_pred_test))
print('The roc_auc_score of the final model:',roc_auc_score(y_test,y_score_test))
print('The confusion matrix of the final model:',confusion_matrix(y_test,y_pred_test))




# Single transaction example. I will use this to test my model when I place it in a server
transaction = [{
    "step": 1,
    "type": "cash_in",
    "amount": 10.964931,
    "oldbalanceorg": 15.04946,
    "newbalanceorig": 15.06615,
    "oldbalancedest": 10.996534,
    "newbalancedest": 0.0
}]

score=pipeline.predict_proba(transaction)[0,1]
if score >=0.038:
    print('This is a fraud transaction')
else:
    print('This is not a fraud transaction')