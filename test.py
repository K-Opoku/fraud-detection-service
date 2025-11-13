import requests
url='http://127.0.0.1:8000/predict'





# Single transaction example. I will use this to test my model when I place it in a server
transaction = {
    "step": 1,
    "type": "cash_in",
    "amount": 10.964931,
    "oldbalanceorg": 15.04946,
    "newbalanceorig": 15.06615,
    "oldbalancedest": 10.996534,
    "newbalancedest": 0.0
}

response=requests.post(url,json=transaction)
prediction=response.json()
print
if prediction['fraud']:
    print('This is a fraud transaction')
else:
    print('This is not a fraud transaction')