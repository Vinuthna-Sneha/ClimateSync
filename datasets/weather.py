import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import sys
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree
from tqdm import tqdm

# Step 1: Initialize Google Earth Engine (GEE)
try:
    ee.Initialize(project='mess-ba866')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='mess-ba866')

# Weather cache for storing retrieved data
class WeatherCache:
    def __init__(self, grid_size=0.25):
        self._cache = {}
        self._grid_size = grid_size

    def get_grid_key(self, lat, lon, date):
        grid_lat = round(lat / self._grid_size) * self._grid_size
        grid_lon = round(lon / self._grid_size) * self._grid_size
        return (grid_lat, grid_lon, date)

    def get(self, lat, lon, date):
        return self._cache.get(self.get_grid_key(lat, lon, date))

    def set(self, lat, lon, date, data):
        self._cache[self.get_grid_key(lat, lon, date)] = data

weather_cache = WeatherCache()
batch_size = 50
max_workers = 5

# Fetch NASA GLDAS weather data
def get_nasa_weather_data(lon, lat, date):
    point = ee.Geometry.Point([lon, lat])

    dataset = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
                .filterBounds(point) \
                .filterDate(f"{date}T00:00:00", f"{date}T23:59:59") \
                .sort('system:time_start', False)

    latest_image = dataset.first()

    weather_bands = latest_image.select([
        'AvgSurfT_inst', 'Wind_f_inst', 'Rainf_f_tavg', 'Qair_f_inst'
    ])

    weather_values = weather_bands.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=5000,
        maxPixels=1e13
    )

    data = weather_values.getInfo()

    if not data:
        return np.nan, np.nan, np.nan, np.nan  # Return NaN if no data is available

    # Fetch data with NoneType handling
    temperature = data.get('AvgSurfT_inst')
    humidity = data.get('Qair_f_inst')
    precipitation = data.get('Rainf_f_tavg')
    wind_speed = data.get('Wind_f_inst')

    # Handle NoneType values before performing calculations
    temperature = (temperature - 273.15) if temperature is not None else np.nan
    humidity = (humidity * 100) if humidity is not None else np.nan
    precipitation = (precipitation * 86400) if precipitation is not None else np.nan
    wind_speed = wind_speed if wind_speed is not None else np.nan

    return temperature, humidity, precipitation, wind_speed
def get_weather_batch(coordinates, start_date, end_date):
    results = []
    date_range = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
    total_tasks = len(coordinates) * len(date_range)
    completed_tasks = 0
    progress_bar = tqdm(total=total_tasks, desc="Fetching Weather Data", unit="data points", dynamic_ncols=True)

    for date in date_range:
        unique_grid_points = {(lat, lon, date) for lat, lon in coordinates if weather_cache.get(lat, lon, date) is None}

        if unique_grid_points:
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_coord = {
                    executor.submit(get_nasa_weather_data, lon, lat, date): (lat, lon, date)
                    for lat, lon, date in unique_grid_points
                }

                for future in as_completed(future_to_coord):
                    lat, lon, date = future_to_coord[future]
                    try:
                        temp, hum, precip, wind = future.result()
                        weather_cache.set(lat, lon, date, {
                            'temperature': temp, 'humidity': hum, 'precipitation': precip, 'wind_speed': wind
                        })
                    except Exception as e:
                        print(f"⚠️ Error fetching weather for {lat}, {lon} on {date}: {e}")

                    completed_tasks += 1
                    progress_bar.update(1)

    for lat, lon in coordinates:
        for date in date_range:
            cached_data = weather_cache.get(lat, lon, date)
            results.append({
                'latitude': lat, 'longitude': lon, 'date': date,
                **(cached_data if cached_data else {'temperature': np.nan, 'humidity': np.nan, 'precipitation': np.nan, 'wind_speed': np.nan})
            })
            completed_tasks += 1
            progress_bar.update(1)

    progress_bar.close()
    return pd.DataFrame(results)

# Impute missing values using nearest valid data points
def impute_missing_values(df, radius_km=1):
    radius_deg = radius_km / 111
    valid_points = df[['latitude', 'longitude']].values
    tree = cKDTree(valid_points)

    for col in ["temperature", "humidity", "precipitation", "wind_speed"]:
        missing_idx = df[df[col].isna()].index

        for idx in missing_idx:
            lat, lon = df.loc[idx, ["latitude", "longitude"]]
            nearby_idx = tree.query_ball_point([lat, lon], r=radius_deg)

            nearby_values = df.loc[nearby_idx, col].dropna().values
            if len(nearby_values) > 0:
                df.loc[idx, col] = np.mean(nearby_values)

    return df

# Retrieve district data and process weather information with progress tracking
def retrieve_district_data(district_name, start_date, end_date, threshold=0.0045, grid_spacing=0.01):
    try:
        ee.Initialize(project='mess-ba866')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='mess-ba866')

    districts = ee.FeatureCollection("FAO/GAUL/2015/level2")
    selected_district = districts.filter(ee.Filter.eq("ADM0_NAME", "India")).filter(ee.Filter.eq("ADM2_NAME", district_name))

    if selected_district.size().getInfo() == 0:
        print(f"❌ No data found for {district_name}. Check spelling.")
        return

    geojson_file = f"{district_name.lower()}_district.geojson"
    geemap.ee_export_vector(selected_district, filename=geojson_file)

    gdf = gpd.read_file(geojson_file)
    gdf["geometry"] = gdf["geometry"].simplify(threshold, preserve_topology=True)
    district_boundary = gdf.geometry.iloc[0]

    minx, miny, maxx, maxy = district_boundary.bounds
    points = [{'latitude': lat, 'longitude': lon} for lat in np.arange(miny, maxy, grid_spacing) for lon in np.arange(minx, maxx, grid_spacing) if district_boundary.contains(Point(lon, lat))]

    print(f"Processing {len(points)} points...")

    coordinates = [(p['latitude'], p['longitude']) for p in points]
    weather_df = get_weather_batch(coordinates, start_date, end_date)
    weather_df = impute_missing_values(weather_df)

    output_file = f"{district_name.lower()}_daily_weather_data_.csv"
    weather_df.to_csv(output_file, index=False)

    print(f"✅ Daily Weather Data saved to {output_file}")
    output_file = f"{district_name.lower()}_daily_weather_data.csv"
    weather_df.to_csv(output_file, index=False)

    print(f"✅ Daily Weather Data saved to {output_file}")

    return weather_df

# Example Usage
if __name__ == "__main__":
    k = retrieve_district_data("Kurnool", start_date="2024-01-01", end_date="2024-2-16")
