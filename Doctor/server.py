from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

# ── Load Environment Variables ──────────────────────────────────────
load_dotenv()

GEOAPIFY_KEY = os.getenv("GEOAPIFY_API_KEY")
POSITIONSTACK_KEY = os.getenv("POSITIONSTACK_API_KEY")

# ── Initialize OpenAI Client (via OpenRouter) ───────────────────────
ai_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("BASE_URL"),  # Must be https://openrouter.ai/api/v1
    default_headers={
        "HTTP-Referer": "https://viora-ai.local",
        "X-Title": "Viora AI Microservice",
    }
)

# ── Initialize FastAPI ──────────────────────────────────────────────
app = FastAPI(title="Independent Doctor Finder API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class DoctorRequest(BaseModel):
    specialist: str
    location: str


# ── Core Functions ──────────────────────────────────────────────────
def geocode_with_fallback(location: str) -> tuple:
    """Attempt geocoding with Positionstack first, fallback to Geoapify."""
    
    # 1. Try Positionstack (Fixed URL encoding for spaces)
    try:
        url = "http://api.positionstack.com/v1/forward"
        params = {"access_key": POSITIONSTACK_KEY, "query": location}
        response = requests.get(url, params=params, timeout=5)
        
        # Log error if key is rejected
        if response.status_code != 200:
            print(f"[!] Positionstack API Error: {response.status_code}")
            
        data = response.json()
        if data.get("data"):
            return data["data"][0]["latitude"], data["data"][0]["longitude"]
    except Exception as e:
        print(f"[!] Positionstack failed: {e}")

    # 2. Fallback to Geoapify
    try:
        url = "https://api.geoapify.com/v1/geocode/search"
        params = {"text": location, "apiKey": GEOAPIFY_KEY}
        response = requests.get(url, params=params, timeout=5)
        features = response.json().get("features", [])
        if features:
            coords = features[0]["geometry"]["coordinates"]
            return coords[1], coords[0]  # lat, lon
    except Exception as e:
        print(f"[!] Geoapify Geocoding failed: {e}")

    return None, None


def normalize_search_term(specialist: str) -> str:
    """Uses a fast AI call to dynamically map specialties to map POI terms."""
    try:
        response = ai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a geocoding assistant. Convert the given medical specialist into a simple, single-word venue type for a map search. Examples: Oncologist -> Cancer, Dermatologist -> Dermatology, General Physician -> Clinic, Pediatrician -> Children. Return ONLY the single word. No punctuation."},
                {"role": "user", "content": specialist}
            ],
            temperature=0, 
            max_tokens=10
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"[!] AI Normalization failed, falling back to original term: {e}")
        return specialist


def search_geoapify(specialist: str, lat: float, lon: float) -> list:
    """Search for clinics using Geoapify, with a broad fallback."""
    search_term = normalize_search_term(specialist)
    print(f"[*] AI Translation: '{specialist}' -> '{search_term}'")
    
    # Format coordinates to exactly 6 decimal places to prevent 400 errors
    clean_lat = f"{float(lat):.6f}"
    clean_lon = f"{float(lon):.6f}"
    
    try:
        url = "https://api.geoapify.com/v2/places"
        
        # --- ATTEMPT 1: Specific AI Term ---
        params_specific = {
            "categories": "healthcare", 
            "filter": f"circle:{clean_lon},{clean_lat},10000",
            "bias": f"proximity:{clean_lon},{clean_lat}",
            "limit": 5,
            "apiKey": GEOAPIFY_KEY,
            "name": search_term # Strict name search
        }
        
        response = requests.get(url, params=params_specific, timeout=15)
        
        if response.status_code != 200:
            print(f"[!] GEOAPIFY ERROR: {response.status_code} - {response.text}")
            results = []
        else:
            results = response.json().get("features", [])
            print(f"[*] Attempt 1 found {len(results)} results.")
            
        # --- ATTEMPT 2: Broad Fallback ---
        # If the strict name search fails, drop the name and find the nearest major medical centers
        if not results:
            print(f"[*] Zero results for '{search_term}'. Triggering broad hospital fallback...")
            params_broad = {
                
                # --- FIX: Change this back to the stable top-level category ---
                "categories": "healthcare", 
                
                "filter": f"circle:{clean_lon},{clean_lat},10000",
                "bias": f"proximity:{clean_lon},{clean_lat}",
                "limit": 5,
                "apiKey": GEOAPIFY_KEY
                # Notice we removed the "name" parameter entirely here!
            }
            
            response_broad = requests.get(url, params=params_broad, timeout=15)
            if response_broad.status_code == 200:
                results = response_broad.json().get("features", [])
                print(f"[*] Fallback found {len(results)} results.")
            else:
                print(f"[!] GEOAPIFY FALLBACK ERROR: {response_broad.status_code}")
                
        # --- PARSE RESULTS ---
        parsed_results = []
        for place in results:
            props = place.get("properties", {})
            parsed_results.append({
                "provider": "Geoapify",
                "name": props.get("name", "N/A"),
                "address": props.get("formatted", "N/A"),
                "lat": place["geometry"]["coordinates"][1],
                "lon": place["geometry"]["coordinates"][0]
            })
        return parsed_results
        
    except Exception as e:
        print(f"[!] CRITICAL Geoapify search crash: {e}")
        return []

# ── API Endpoint ────────────────────────────────────────────────────
@app.post("/find-doctors")
async def find_doctors_endpoint(req: DoctorRequest):
    if not req.specialist or not req.location:
        raise HTTPException(status_code=400, detail="Both specialist and location are required.")

    # 1. Geocode the location
    lat, lon = geocode_with_fallback(req.location)
    if lat is None:
        raise HTTPException(status_code=404, detail=f"Could not find coordinates for location: {req.location}")

    # 2. Search for doctors (Using Geoapify)
    results = search_geoapify(req.specialist, lat, lon)

    if not results:
        return {"doctors": [], "message": f"No {req.specialist} found in {req.location}."}

    return {"doctors": results, "message": f"Found {len(results)} results."}