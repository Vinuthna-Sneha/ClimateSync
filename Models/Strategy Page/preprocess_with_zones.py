import csv
import json
from io import StringIO
import pandas as pd
import googlemaps

# Initialize Google Maps API client
API_KEY = "AIzaSyCUS9uEV_iQHG_I3BofUVN4bPtOA5L6P0E"  # Replace with your actual API key
gmaps = googlemaps.Client(key=API_KEY)

# Load input data
doc1 = pd.read_csv("updated_data_with_cri_sklm.csv")  # Primary CSV with full data
doc2 = pd.read_csv("shap_insights_sklm.csv")  # Trend data
doc4 = pd.read_csv("sklm_zone_wise_analysis.csv")  # Secondary CSV for comparison

file_path = 'output.json'
with open(file_path, 'r') as file:
    doc3 = json.load(file)  # JSON with socioeconomic and industry data

# Function to parse CSV-like data from a DataFrame
def parse_csv_data(df):
    csv_string = df.to_csv(index=False)
    f = StringIO(csv_string)
    reader = csv.DictReader(f)
    return {int(row['zone']): row for row in reader}

# Function to parse trend data from a DataFrame
def parse_trend_data(df):
    csv_string = df.to_csv(index=False)
    f = StringIO(csv_string)
    reader = csv.DictReader(f)
    trends = {}
    for row in reader:
        zone = int(row['Zone'])
        if zone not in trends:
            trends[zone] = []
        trends[zone].append({
            'trend_description': row['Trend_Description'],
            'influencing_feature': row['Influencing_Feature'],
            'impact': float(row['Impact']),
            'effect': row['Effect']
        })
    return trends

# Function to parse JSON data (already loaded as a Python object)
def parse_json_data(data):
    return {(item['location']['latitude'], item['location']['longitude']): item for item in data}

# Function to get a zone name from coordinates using Google Maps API
def get_famous_place_from_coords(lat, lon, zone):
    try:
        reverse_geocode_result = gmaps.reverse_geocode((lat, lon))
        if reverse_geocode_result:
            # Prioritize locality (city/town) and sublocality (district)
            for component in reverse_geocode_result[0]['address_components']:
                if "locality" in component["types"]:
                    return component["long_name"]
                if "sublocality" in component["types"]:
                    return component["long_name"]
            # Default to formatted address if no locality is found
            return reverse_geocode_result[0]['formatted_address']
    except Exception as e:
        print(f"Google Maps API error for ({lat}, {lon}): {e}")
        return f"Zone {zone}"

# Parse all documents
data1 = parse_csv_data(doc1)  # Updated CRI data
data2 = parse_trend_data(doc2)  # SHAP insights
data3 = parse_json_data(doc3)  # Socioeconomic and industry data
data4 = parse_csv_data(doc4)  # Zone analysis final

# Unified structure
unified_data = []

# Merge data
for zone in data1.keys():
    zone_data = data1[zone]
    lat, lon = float(zone_data['latitude']), float(zone_data['longitude'])
    
    # Get the unique name based on coordinates
    unique_name = get_famous_place_from_coords(lat, lon, zone)
    
    # Initialize unified zone entry
    unified_zone = {
        'zone': zone,
        'zone_name': unique_name,  # Add the unique name here
        'location': {
            'latitude': lat,
            'longitude': lon
        },
        'land_cover': {
            'bare': float(zone_data['bare']),
            'built': float(zone_data['built']),
            'crops': float(zone_data['crops']),
            'flooded_vegetation': float(zone_data['flooded_vegetation']),
            'grass': float(zone_data['grass']),
            'shrub_and_scrub': float(zone_data['shrub_and_scrub']),
            'snow_and_ice': float(zone_data['snow_and_ice']),
            'trees': float(zone_data['trees']),
            'water': float(zone_data['water'])
        },
        'environmental_factors': {
            'temperature': float(zone_data['temperature']),
            'humidity': float(zone_data['humidity']),
            'precipitation': float(zone_data['precipitation']),
            'wind_speed': float(zone_data['wind_speed']),
            'BCCMASS': float(zone_data['BCCMASS']),
            'CH4_column_volume_mixing_ratio_dry_air_bias_corrected': float(zone_data['CH4_column_volume_mixing_ratio_dry_air_bias_corrected']),
            'CO_column_number_density': float(zone_data['CO_column_number_density']),
            'DUCMASS': float(zone_data['DUCMASS']),
            'NO2_column_number_density': float(zone_data['NO2_column_number_density']),
            'O3_column_number_density': float(zone_data['O3_column_number_density']),
            'SO2_column_number_density': float(zone_data['SO2_column_number_density']),
            'absorbing_aerosol_index': float(zone_data['absorbing_aerosol_index']),
            'tropospheric_HCHO_column_number_density': float(zone_data['tropospheric_HCHO_column_number_density'])
        },
        'risk_metrics': {
            'Health_Risk_Index': float(zone_data['Health_Risk_Index']),
            'Urban_Heat_Index': float(zone_data['Urban_Heat_Index']),
            'Real_Estate_Risk': float(zone_data['Real_Estate_Risk']),
            'Green_Score': float(zone_data['Green_Score']),
            'H(Hazard)': float(zone_data['H']),
            'E(Exposure)': float(zone_data['E']),
            'V(Vulnerability)': float(zone_data['V']),
            'CRI': float(zone_data['CRI']),
            'Risk_Category': zone_data['Risk_Category']
        },
        'trends': data2.get(zone, []),
        'socioeconomic_status': {},
        'industries_present': [],
        'research_findings': {},
        'web_search': []
    }
    
    # Merge JSON data (doc3) if available
    if (lat, lon) in data3:
        json_data = data3[(lat, lon)]
        unified_zone['socioeconomic_status'] = json_data['socioeconomic_status']
        unified_zone['industries_present'] = json_data['industries_present']
        unified_zone['research_findings'] = json_data['research_findings']
        unified_zone['web_search'] = json_data['web_search']
    
    # Compare with doc4 and log differences
    if zone in data4:
        for key, value in data4[zone].items():
            if key not in ['zone', 'latitude', 'longitude'] and key in zone_data:
                if float(zone_data[key]) != float(value):  # Compare as floats
                    print(f"Warning: Differing data for zone {zone}, key {key}: {zone_data[key]} vs {value}")

    unified_data.append(unified_zone)

# Convert to JSON and save
with open('unified_data.json', 'w') as f:
    json.dump(unified_data, f, indent=2)

print("Unified data saved to 'unified_data.json'")
