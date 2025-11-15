import requests
url='http://127.0.0.1:8000/predict'





# Single transaction example. I will use this to test my model when I place it in a server
transaction = {
  "step": 1,
  "type": "cash_in",
  "amount": 57809.81,
  "oldbalanceorg": 3434765.61,
  "newbalanceorig": 3492575.42,
  "oldbalancedest": 59666.0,
  "newbalancedest": 0.0
}


response=requests.post(url,json=transaction)
prediction=response.json()
print
if prediction['fraud']:
    print('This is a fraud transaction')
else:
    print('This is not a fraud transaction')