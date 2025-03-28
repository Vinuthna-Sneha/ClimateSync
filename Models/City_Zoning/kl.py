import requests
import json
import pandas as pd
from scholarly import scholarly
from multiprocessing import Pool

# Google API Configuration
GOOGLE_SEARCH_API_KEY = "AIzaSyCUS9uEV_iQHG_I3BofUVN4bPtOA5L6P0E"
SEARCH_ENGINE_ID = "85b91bc18405b46e4"

# ---------------------------
# 1. Read CSV and Process Data
# ---------------------------
def read_csv_and_process(file_path):
    df = pd.read_csv(file_path)  # Assuming CSV has 'latitude' and 'longitude' columns
    knowledge_data = []

    for index, row in df.iterrows():
        lat, lng = row["latitude"], row["longitude"]
        print(f"Processing {index + 1}: Latitude {lat}, Longitude {lng}")
        knowledge_layer = build_knowledge_layer(lat, lng)
        knowledge_data.append(knowledge_layer)
    
    # Save results as JSON
    with open("output.json", "w") as f:
        json.dump(knowledge_data, f, indent=4)

    print("Processing complete. Data saved to output.json")

# ---------------------------
# 2. Socio-Economic Data
# ---------------------------
def get_socioeconomic_data(latitude, longitude):
    api_key = "AIzaSyCUS9uEV_iQHG_I3BofUVN4bPtOA5L6P0E"
    types = ["school", "hospital", "bank", "grocery_store"]
    summary = {"schools": 0, "hospitals": 0, "banks": 0, "grocery_stores": 0}

    for place_type in types:
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius=5000&type={place_type}&key={api_key}"
        response = requests.get(url).json()
        if response.get("status") == "OK":
            summary[place_type + "s"] = len(response["results"])
    
    return summary

# ---------------------------
# 3. Industries Data
# ---------------------------
def get_industry_data(latitude, longitude, radius=10000):
    api_key = "AIzaSyCUS9uEV_iQHG_I3BofUVN4bPtOA5L6P0E"
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius={radius}&type=industrial_establishment&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return [{"name": place.get("name"), "address": place.get("vicinity"),"types": place.get("types")} for place in data.get("results", [])]
    return {"error": f"Status code {response.status_code}"}

# ---------------------------
# 4. Research Data
# ---------------------------
def get_research_and_survey_data(latitude, longitude):
    """
    Fetches research papers, surveys, and reports related to:
    - Water Management
    - Soil Status
    - Population
    
    Uses Google Scholar (for academic papers) and Web Search (for general reports).
    """
    queries = ["water management", "soil status", "population"]
    research_results = {}

    for query in queries:
        search_query = f"{query} near {latitude}, {longitude}"
        research_items = []
        survey_items = []

        # Fetch academic research papers
        try:
            search_results = scholarly.search_pubs(search_query)
            for _ in range(3):  # Limit to 3 papers per query
                pub = next(search_results, None)
                if pub:
                    research_items.append({
                        "title": pub.get("bib", {}).get("title"),
                        "author": pub.get("bib", {}).get("author"),
                        "venue": pub.get("bib", {}).get("venue"),
                        "year": pub.get("bib", {}).get("pub_year")
                    })
        except Exception as e:
            research_items = {"error": str(e)}

        # Fetch general surveys and reports from the internet
        survey_items = search_web(f"{query} site:.gov OR site:.org OR site:.edu OR survey OR report")

        research_results[query] = {
            "research_papers": research_items,
            "surveys_and_reports": survey_items
        }

    return research_results

# ---------------------------
# 5. Web Search (Using Google Custom Search API)
# ---------------------------
def search_web(query):
    """Fetch only website links using Google Custom Search API."""
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_SEARCH_API_KEY}&cx={SEARCH_ENGINE_ID}"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if "items" in data:
            return [item.get("link") for item in data["items"][:5]]  # Extract only URLs
        
        return []  # Return empty if no results found
    except Exception as e:
        return [f"Error: {str(e)}"]

# ---------------------------
# Function: Parallel Web Search Execution
# ---------------------------
def parallel_search(queries):
    """Runs multiple queries in parallel and returns links."""
    with Pool(processes=3) as pool:  # Run 3 searches in parallel
        results = pool.map(search_web, queries)
    
    return {queries[i]: results[i] for i in range(len(queries))}


# ---------------------------
# 6. Build Knowledge Layer
# ---------------------------
def build_knowledge_layer(lat, lng):
    return {
        "location": {"latitude": lat, "longitude": lng},
        "socioeconomic_status": get_socioeconomic_data(lat, lng),
        "industries_present": get_industry_data(lat, lng),
        "research_findings": get_research_and_survey_data(lat, lng),
        "web_search": search_web(f"latest water pollution survey site:.gov OR site:.org OR site:.edu for latitude {lat}, longitude {lng}")
    }

# ---------------------------
# 7. Run the Script
# ---------------------------
if __name__ == "__main__":
    csv_file = "updated_data_with_cri_sklm.csv"  # Update with actual file path
    read_csv_and_process(csv_file)
