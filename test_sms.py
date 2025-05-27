import requests
from requests.auth import HTTPBasicAuth

url = "          "
username = "sms"
password = "  "

payload = {
    "message": "có người ngã ở phòng khách",
    "phoneNumbers": ["number phone"],
}

response = requests.post(url, json=payload, auth=HTTPBasicAuth(username, password))

print("Response:", response)