import requests

url = "http://localhost:5000/health"

print("Requesting health check...")
r = requests.get(url)
print(f"Response: {r.json()}")
