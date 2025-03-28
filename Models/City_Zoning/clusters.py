# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import folium
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Polygon
import matplotlib.cm as cm
import json  # Added for JSON output

# ---------------------------
# 1. Load Datasets (Update paths as needed)
# ---------------------------
biweekly_df = pd.read_csv("/content/krishna_biweekly_next.csv")
monthly_df = pd.read_csv("/content/krishna_monthly_next.csv")
daily_df = pd.read_csv("/content/krishna_daily_weather_data.csv")
weekly_df = pd.read_csv("/content/krishna_weekly_next.csv")
# Drop unnecessary columns
# biweekly_df = biweekly_df.drop(columns=['label', 'snow_and_ice'])
# monthly_df = monthly_df.drop(columns=['label', 'snow_and_ice'])
daily_df_orig = daily_df.copy()  # Keep original for trends
daily_df = daily_df.drop(columns=['date'])
weekly_df = weekly_df.dropna()

# Compute temporal trends from daily data
daily_trends = daily_df_orig.groupby(['latitude', 'longitude']).agg({
    'temperature': lambda x: np.polyfit(range(len(x)), x, 1)[0],  # Slope of temperature
    'precipitation': 'std'  # Variability of precipitation
}).reset_index().rename(columns={'temperature': 'temp_trend', 'precipitation': 'precip_std'})

# ---------------------------
# 2. Merge Datasets
# ---------------------------
merged_df = biweekly_df.merge(monthly_df, on=["latitude", "longitude"], suffixes=('_biweekly', '_monthly')) \
                       .merge(daily_df, on=["latitude", "longitude"]) \
                       .merge(weekly_df, on=["latitude", "longitude"]) \
                       .merge(daily_trends, on=["latitude", "longitude"])

merged_df = merged_df.drop_duplicates(subset=['latitude', 'longitude'])
imputer = KNNImputer(n_neighbors=5)
merged_df = pd.DataFrame(imputer.fit_transform(merged_df), columns=merged_df.columns)

# ---------------------------
# 3. Feature Preparation with Comprehensive Weighting
# ---------------------------
feature_weights = {
    'built_biweekly': 1.5, 'trees_biweekly': 1.3, 'flooded_vegetation_biweekly': 1.8,
    'CO_column_number_density': 1.5, 'NO2_column_number_density': 1.5,
    'temperature': 1.2, 'precipitation': 1.4, 'temp_trend': 1.3, 'precip_std': 1.2
}
non_spatial_features = merged_df.drop(columns=['latitude', 'longitude'])
for col in non_spatial_features.columns:
    if col in feature_weights:
        non_spatial_features[col] *= feature_weights[col]

# Spatial features: Use MinMaxScaler to preserve coordinate differences (Manhattan-like)
spatial_features = merged_df[['latitude', 'longitude']]
scaler_spatial = MinMaxScaler()
scaled_spatial = scaler_spatial.fit_transform(spatial_features)

# Non-spatial features: Use MinMaxScaler instead of StandardScaler for Manhattan-like scaling
scaler_non_spatial = MinMaxScaler()
scaled_non_spatial = scaler_non_spatial.fit_transform(non_spatial_features)

# Combine features
scaled_features = np.hstack([scaled_spatial, scaled_non_spatial])

# ---------------------------
# 4. Determine Optimal Number of Clusters
# ---------------------------
K_range = range(5, 16)  # Limited range for magistrate manageability
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    silhouette_scores.append(silhouette_score(scaled_features, labels))

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal Clusters')
plt.xlabel('Number of Zones')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

optimal_k = K_range[np.argmax(silhouette_scores)]
if optimal_k > 15:  # Cap for manageability
    optimal_k = 15
print(f"Selected optimal_k: {optimal_k}")

# ---------------------------
# 5. Apply K-Means Clustering
# ---------------------------
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
merged_df['zone'] = kmeans.fit_predict(scaled_features)

print("Number of points per zone:")
print(merged_df['zone'].value_counts())

# ---------------------------
# 6. Create Voronoi Diagram (No Subdivisions)
# ---------------------------
centroids_scaled = kmeans.cluster_centers_[:, :2]
centroids = scaler_spatial.inverse_transform(centroids_scaled)
vor = Voronoi(centroids)

# Use convex hull as district boundary (no shapefile)
points = MultiPoint(list(zip(merged_df['longitude'], merged_df['latitude'])))
district_boundary = points.convex_hull
print("Using convex hull as district boundary.")

def assign_voronoi_region(point, vor, centroids):
    point_scaled = scaler_spatial.transform([point])[0]
    dists = np.array([np.linalg.norm(point_scaled - centroid) for centroid in centroids_scaled])
    return np.argmin(dists)

# Assign Voronoi zones
merged_df['voronoi_zone'] = [assign_voronoi_region([row['latitude'], row['longitude']], vor, centroids)
                             for _, row in merged_df.iterrows()]

print("Number of points per Voronoi zone:")
print(merged_df['voronoi_zone'].value_counts())

# ---------------------------
# 7. Advanced Zone Classification
# ---------------------------
def classify_zone(row):
    flood_risk = row['flooded_vegetation_biweekly'] + row['precipitation'] * 0.1
    air_quality = row['CO_column_number_density'] + row['NO2_column_number_density']
    if row['built_biweekly'] > 0.6 and air_quality > 0.0002:
        return "Urban High-Pollution"
    elif flood_risk > 0.5:
        return "Flood-Prone Resilience"
    elif row['trees_biweekly'] > 0.5:
        return "Green Buffer"
    else:
        return "Mixed Low-Impact"

zone_summary = merged_df.groupby('voronoi_zone').mean()
zone_summary['zone_type'] = zone_summary.apply(classify_zone, axis=1)

# ---------------------------
# 8. Enhanced Visualization
# ---------------------------
map_center = [merged_df['latitude'].mean(), merged_df['longitude'].mean()]
zone_map = folium.Map(location=map_center, zoom_start=10)

# Create a unique color for each voronoi_zone
unique_zones = merged_df['voronoi_zone'].unique()
colormap = cm.get_cmap('tab20', len(unique_zones))
colors = [colormap(i) for i in range(len(unique_zones))]
colors_hex = [f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for c in colors]
zone_color_map = dict(zip(unique_zones, colors_hex))  # Map each zone to its color

# Plot Voronoi polygons with consistent zone colors
for zone in unique_zones:
    region_idx = vor.point_region[zone]
    if region_idx < len(vor.regions):
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            polygon_points = [vor.vertices[i] for i in region]
            voronoi_poly = Polygon(polygon_points)
            clipped_poly = voronoi_poly.intersection(district_boundary)
            if clipped_poly.is_empty or clipped_poly.geom_type not in ['Polygon', 'MultiPolygon']:
                continue
            if clipped_poly.geom_type == 'Polygon':
                clipped_coords = list(clipped_poly.exterior.coords)
            else:
                clipped_coords = list(clipped_poly.geoms[0].exterior.coords)

            popup_text = (f"Zone {zone}: {zone_summary.loc[zone, 'zone_type']}<br>"
                          f"Temp: {zone_summary.loc[zone, 'temperature']:.1f}°C<br>"
                          f"Precip: {zone_summary.loc[zone, 'precipitation']:.1f} mm<br>"
                          f"CO: {zone_summary.loc[zone, 'CO_column_number_density']:.4f}<br>"
                          f"NO2: {zone_summary.loc[zone, 'NO2_column_number_density']:.4f}")
            folium.Polygon(
                locations=[[coord[1], coord[0]] for coord in clipped_coords],
                color=zone_color_map[zone],
                fill=True,
                fill_color=zone_color_map[zone],
                fill_opacity=0.4,
                weight=2,
                popup=popup_text
            ).add_to(zone_map)

# Plot all coordinates with their assigned zone colors
for _, row in merged_df.iterrows():
    zone = row['voronoi_zone']
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color=zone_color_map[zone],
        fill=True,
        fill_opacity=0.7,
        popup=f"Zone {zone}"
    ).add_to(zone_map)

# Add a legend (optional, manual implementation)
legend_html = """
     <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; font-size:14px;
     background-color:white; padding: 10px; border: 2px solid grey;">
     <b>Zone Types</b><br>
     <i style='background:#FF0000; width:10px; height:10px; display:inline-block;'></i> Urban High-Pollution<br>
     <i style='background:#00FF00; width:10px; height:10px; display:inline-block;'></i> Green Buffer<br>
     <i style='background:#0000FF; width:10px; height:10px; display:inline-block;'></i> Flood-Prone Resilience<br>
     <i style='background:#808080; width:10px; height:10px; display:inline-block;'></i> Mixed Low-Impact<br>
     </div>
"""
zone_map.get_root().html.add_child(folium.Element(legend_html))

zone_map.save("krishna_voronoi_zoning_map.html")

# ---------------------------
# 9. Save Outputs
# ---------------------------
zone_summary.to_csv("krishna_voronoi_zone_summary.csv")

# Generate JSON with zones and their coordinates
zone_coords = {}
for zone in merged_df['voronoi_zone'].unique():
    zone_data = merged_df[merged_df['voronoi_zone'] == zone][['latitude', 'longitude']].to_dict(orient='records')
    zone_coords[str(zone)] = zone_data  # Convert zone to string for JSON compatibility

# Save to JSON file
with open("krishna_zone_coordinates.json", "w") as json_file:
    json.dump(zone_coords, json_file, indent=4)

print(f"Created {len(merged_df['voronoi_zone'].unique())} zones. "
      f"Map saved as 'srikakulam_voronoi_zoning_map.html'. "
      f"Summary saved as 'srikakulam_voronoi_zone_summary.csv'. "
      f"Zone coordinates saved as 'srikakulam_zone_coordinates.json'.")
# ---------------------------
# 1. Column Mapping (Restore Original Column Names)
# ---------------------------
column_mapping = {
    'bare_biweekly': 'bare', 'built_biweekly': 'built', 'crops_biweekly': 'crops',
    'flooded_vegetation_biweekly': 'flooded_vegetation', 'grass_biweekly': 'grass',
    'shrub_and_scrub_biweekly': 'shrub_and_scrub', 'trees_biweekly': 'trees',
    'water_biweekly': 'water', 'snow_and_ice_biweekly': 'snow_and_ice'  # Add back snow_and_ice
}

merged_df = merged_df.rename(columns=column_mapping)

# ---------------------------
# 2. Select and Reorder Final Columns
# ---------------------------
final_columns = [
    'latitude', 'longitude', 'bare', 'built', 'crops', 'flooded_vegetation', 'grass', 'shrub_and_scrub',
    'snow_and_ice', 'trees', 'water', 'bare_monthly', 'built_monthly', 'crops_monthly',
    'flooded_vegetation_monthly', 'grass_monthly', 'shrub_and_scrub_monthly', 'trees_monthly',
    'water_monthly', 'temperature', 'humidity', 'precipitation', 'wind_speed', 'BCCMASS',
    'CH4_column_volume_mixing_ratio_dry_air_bias_corrected', 'CO_column_number_density', 'DUCMASS',
    'NO2_column_number_density', 'O3_column_number_density', 'SO2_column_number_density',
    'absorbing_aerosol_index', 'tropospheric_HCHO_column_number_density', 'zone'
]

# Select and reorder columns
full_df = merged_df[final_columns]

# ---------------------------
# 3. Save Final Dataset to CSV
# ---------------------------
full_df.to_csv("krishna_full_dataset_with_zones.csv", index=False)

print(f"✅ Final dataset saved as 'srikakulam_full_dataset_with_zones.csv'.")