import requests

url = "http://localhost:5000/invocations"

data = {"p_class": 3, "sex": "male", "age": 28}

headers = {"Content-Type": "application/json"}
print("Sending request for model...")
print(f"Data: {data}")
r = requests.post(url, json=data, headers=headers)
print(f"Response: {r.text}")
