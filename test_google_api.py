import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.gentenv("API_KEY")

#Example: Airport to various destinations
origin = "Edinburgh Airport"
destinations = [
    "12 Queen St, Edinburgh",
    "9 King St, Edinburgh",
    "5 Prince St, Edinburgh",
    "20 George St, Edinburgh",
    "1 Victoria St, Edinburgh"
]

url = "https://maps.googleapis.com/maps/api/distancematrix/json"

params = {
    "origins:": origin,
    "destinations": "|".join(destinations),
    "mode": "driving",
    "key": API_KEY
}

response = requests.get(url, params=params)
data = response.json()

print("Request URL:", requests.Request('GET', url, params=params).prepare().url)

print("API Response:")
print(data)

