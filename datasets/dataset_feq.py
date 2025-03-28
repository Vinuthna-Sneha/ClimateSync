import ee
import geemap
import geopandas as gpd
import pandas as pd
import json
import numpy as np
import requests
import time
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
try:
    ee.Initialize(project='vertical-setup-450217-n2')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='vertical-setup-450217-n2')



# Initialize global variables
district_cache = {}

batch_size = 50
max_workers = 5

def initialize_datasets(start_date, end_date):
    """Initialize and filter datasets based on date range"""
    datasets = {
        # Daily datasets


        # 7-day interval datasets
        'no2': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
            .select('NO2_column_number_density'),
        'so2': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2')
            .select('SO2_column_number_density'),
        'ch4': ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4')
            .select('CH4_column_volume_mixing_ratio_dry_air_bias_corrected'),
        'aerosol': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI')
            .select('absorbing_aerosol_index'),
        'hcho': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_HCHO')
            .select('tropospheric_HCHO_column_number_density'),
        'merra': ee.ImageCollection('NASA/GSFC/MERRA/aer/2')
            .select(['DUCMASS', 'BCCMASS']),
       'co': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO').select('CO_column_number_density'),
            'ozone': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3').select('O3_column_number_density'),

        # 15-day interval datasets
        'landcover': ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1'),

        # Monthly datasets
        'monthly_landcover': ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    }

    # Filter date ranges for all datasets
    for key in datasets:
        if datasets[key] is not None:
            datasets[key] = datasets[key].filterDate(start_date, end_date)

    return datasets



def process_single_point(point, datasets, start_date, end_date):
    """Process data for a single point with different temporal frequencies"""
    lat, lon = point['latitude'], point['longitude']
    ee_point = ee.Geometry.Point([lon, lat])

    try:
        # Daily data (weather) is handled separately in get_weather_batch

        # 7-day interval data
        weekly_data = ee.Image.cat([
            datasets['no2'].mean(),
            datasets['so2'].mean(),
            datasets['ch4'].mean(),
            datasets['aerosol'].mean(),
            datasets['hcho'].mean(),
            datasets['merra'].mean(),
            datasets['co'].mean(),
            datasets['ozone'].mean(),
        ]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_point,
            scale=10000,
            bestEffort=True
        ).getInfo()

        # 15-day interval data
        biweekly_data = datasets['landcover'].median().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_point,
            scale=10000,
            bestEffort=True
        ).getInfo()

        # Monthly data
        monthly_data = datasets['monthly_landcover'].median().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_point,
            scale=10000,
            bestEffort=True
        ).getInfo()

        return {
            'latitude': lat,
            'longitude': lon,
            'weekly_data': weekly_data,
            'biweekly_data': biweekly_data,
            'monthly_data': monthly_data
        }

    except Exception as e:
        print(f"Error processing point ({lat}, {lon}): {e}")
        return None

def retrieve_district_data(district_name, start_date=None, end_date=None, threshold=0.001, grid_spacing=0.01):
    """Main function to retrieve district data with different temporal frequencies"""

    # Set default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()

    # Convert dates to ee.Date objects
    ee_start_date = ee.Date(start_date.strftime('%Y-%m-%d'))
    ee_end_date = ee.Date(end_date.strftime('%Y-%m-%d'))

    try:
        ee.Initialize(project='vertical-setup-450217-n2')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='vertical-setup-450217-n2')

    # Load and process district boundary
    districts = ee.FeatureCollection("FAO/GAUL/2015/level2")
    india_districts = districts.filter(ee.Filter.eq("ADM0_NAME", "India"))
    selected_district = india_districts.filter(ee.Filter.eq("ADM2_NAME", district_name))

    if selected_district.size().getInfo() == 0:
        print(f"❌ No data found for {district_name}. Check spelling.")
        return

    # Export and process district geometry
    geojson_file = f"{district_name.lower()}_district.geojson"
    geemap.ee_export_vector(selected_district, filename=geojson_file)

    gdf = gpd.read_file(geojson_file)
    gdf["geometry"] = gdf["geometry"].simplify(threshold, preserve_topology=True)
    district_boundary = gdf.geometry.iloc[0]

    # Initialize datasets
    datasets = initialize_datasets(ee_start_date, ee_end_date)

    # Generate grid points
    minx, miny, maxx, maxy = district_boundary.bounds
    points = []
    for lat in np.arange(miny, maxy, grid_spacing):
        for lon in np.arange(minx, maxx, grid_spacing):
            point = Point(lon, lat)
            if district_boundary.contains(point):
                points.append({'latitude': lat, 'longitude': lon})

    # Process points in batches
    point_batches = [points[i:i + batch_size] for i in range(0, len(points), batch_size)]
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch in point_batches:
            # Get weather data for batch
            coordinates = [(p['latitude'], p['longitude']) for p in batch]


            # Process Earth Engine data for batch
            batch_results = []
            for point in batch:
                point_data = process_single_point(point, datasets, start_date, end_date)
                if point_data:
                    # Add weather data
                    point_data['daily_weather'] =[]
                    batch_results.append(point_data)

            all_results.extend(batch_results)


    weekly_data = []
    biweekly_data = []
    monthly_data = []

    for result in all_results:

        # Process weekly data
        weekly_data.append({
            'latitude': result['latitude'],
            'longitude': result['longitude'],
            **result['weekly_data']
        })

        # Process biweekly data
        biweekly_data.append({
            'latitude': result['latitude'],
            'longitude': result['longitude'],
            **result['biweekly_data']
        })

        # Process monthly data
        monthly_data.append({
            'latitude': result['latitude'],
            'longitude': result['longitude'],
            **result['monthly_data']
        })

    # Save to separate CSV files
    base_filename = f"{district_name.lower()}"
    pd.DataFrame(weekly_data).to_csv(f"/content/drive/MyDrive/{base_filename}_weekly_next.csv", index=False)
    pd.DataFrame(biweekly_data).to_csv(f"/content/drive/MyDrive/{base_filename}_biweekly_next.csv", index=False)
    pd.DataFrame(monthly_data).to_csv(f"/content/drive/MyDrive/{base_filename}_monthly_next.csv", index=False)

    print(f"✅ Data collection complete. Files saved with prefix: {base_filename}")
    return {

        'weekly': pd.DataFrame(weekly_data),
        'biweekly': pd.DataFrame(biweekly_data),
        'monthly': pd.DataFrame(monthly_data)
    }

if __name__ == "__main__":
    # Example usage with date range
    start_date = datetime(2024, 2, 17)
    end_date = datetime(2024, 12, 31)
    retrieve_district_data("Kurnool", start_date, end_date, threshold=0.0045)