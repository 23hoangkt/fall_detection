import requests
from requests.auth import HTTPBasicAuth

url = "http://192.168.61.19:8080/message"
username = "sms"
password = "xtV-wKFL"

payload = {
    "message": "có lửa",
    "phoneNumbers": ["+84325372909"],
}

response = requests.post(url, json=payload, auth=HTTPBasicAuth(username, password))

print("Response:", response)