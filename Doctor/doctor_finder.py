import requests
import os
from dotenv import load_dotenv

load_dotenv()

GEOAPIFY_KEY = os.getenv("GEOAPIFY_API_KEY")

def geocode_location(location: str) -> tuple:
    url = "https://api.geoapify.com/v1/geocode/search"
    params = {"text": location, "apiKey": GEOAPIFY_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    features = data.get("features", [])
    if not features:
        return None, None

    coords = features[0]["geometry"]["coordinates"]
    return coords[1], coords[0]  # lat, lon


def find_doctors(specialist: str, location: str) -> list:
    lat, lon = geocode_location(location)
    if lat is None:
        return []

    url = "https://api.geoapify.com/v2/places"
    params = {
        "categories": "healthcare.doctor,healthcare.clinic",
        "filter": f"circle:{lon},{lat},10000",
        "bias": f"proximity:{lon},{lat}",
        "limit": 5,
        "apiKey": GEOAPIFY_KEY,
        "name": specialist
    }

    response = requests.get(url, params=params)
    features = response.json().get("features", [])

    results = []
    for place in features:
        props = place.get("properties", {})
        results.append({
            "name"   : props.get("name", "N/A"),
            "address": props.get("formatted", "N/A"),
            "phone"  : props.get("contact", {}).get("phone", "Not listed"),
            "website": props.get("website", "Not listed"),
            "lat"    : place["geometry"]["coordinates"][1],
            "lon"    : place["geometry"]["coordinates"][0]
        })

    return results