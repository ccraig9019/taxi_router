import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

#Example: Airport to various destinations
points = [
    "Edinburgh Airport",
    "12 Queen St, Edinburgh",
    "9 King St, Edinburgh",
    "5 Prince St, Edinburgh",
    "20 George St, Edinburgh",
    "1 Victoria St, Edinburgh"
]

url = "https://maps.googleapis.com/maps/api/distancematrix/json"

params = {
    "origins": "|".join(points),
    "destinations": "|".join(points),
    "mode": "driving",
    "key": API_KEY
}

response = requests.get(url, params=params)
data = response.json()

#print("Request URL:", requests.Request('GET', url, params=params).prepare().url)
print("Distance Matrix (in km):")
header = ["From/To"] + points
print("{:<25}".format(header[0]), end="")
for h in header[1:]:
    print("{:<25}".format(h), end="")
print()

#print each row
for i, row in enumerate(data["rows"]):
    print("{:<25}".format(points[i]), end="")
    for elem in row["elements"]:
        if elem["status"] == "OK":
            dist_km = elem["distance"]["value"] / 1000  # convert meters to km
            print("{:<25}".format(f"{dist_km:.1f} km"), end="")
        else:
            print("{:<25}".format("N/A"), end="")
    print()
