import requests
from requests.auth import HTTPBasicAuth

url = "http://192.168.1.5:8080/message"
username = "sms"
password = "kLUPTcJ_"

payload = {
    "message": "có người ngã ở phòng khách",
    "phoneNumbers": ["+84974039288"],
}

response = requests.post(url, json=payload, auth=HTTPBasicAuth(username, password))

print("Response:", response)