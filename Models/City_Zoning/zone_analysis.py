import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Load Dataset
file_path = "srikakulam_full_dataset_with_zones.csv"  # Update with actual path
df = pd.read_csv(file_path)

# Drop missing values
df.dropna(inplace=True)

# **Step 1: Normalize Features**
scaler = MinMaxScaler()
features_to_scale = [
    "NO2_column_number_density", "SO2_column_number_density",
    "CH4_column_volume_mixing_ratio_dry_air_bias_corrected",
    "absorbing_aerosol_index", "tropospheric_HCHO_column_number_density",
    "temperature", "humidity", "precipitation", "wind_speed",
    "built", "bare", "crops", "flooded_vegetation", "grass", "shrub_and_scrub",
    "snow_and_ice", "trees", "water", "BCCMASS", "CO_column_number_density", "DUCMASS",
    "O3_column_number_density"
]
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# **Step 2: Clustering into 10-15 Zones**
num_clusters = 15  # Adjust based on results
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['zone'] = kmeans.fit_predict(df[['latitude', 'longitude']])

# **Step 3: Compute Impact Indices**
df["Health_Risk_Index"] = (
    df["NO2_column_number_density"] * 0.4 +
    df["SO2_column_number_density"] * 0.3 +
    df["CH4_column_volume_mixing_ratio_dry_air_bias_corrected"] * 0.2 +
    df["absorbing_aerosol_index"] * 0.1
)

df["Urban_Heat_Index"] = (
    df["temperature"] +
    df["NO2_column_number_density"] * 2 +
    df["SO2_column_number_density"] * 1.5 +
    df["built"] * 3
)

df["Real_Estate_Risk"] = (
    df["NO2_column_number_density"] * 0.5 +
    df["SO2_column_number_density"] * 0.3 +
    df["built"] * 0.2
)

df["Green_Score"] = (
    df["trees"] * 0.5 +
    df["grass"] * 0.3 +
    df["shrub_and_scrub"] * 0.2
)

# **Step 4: Compute Zone-Wise Aggregates (Mean Values + Risks)**
zone_data = df.groupby("zone").agg({
    "latitude": "mean",
    "longitude": "mean",
    "bare": "mean",
    "built": "mean",
    "crops": "mean",
    "flooded_vegetation": "mean",
    "grass": "mean",
    "shrub_and_scrub": "mean",
    "snow_and_ice": "mean",
    "trees": "mean",
    "water": "mean",
    "temperature": "mean",
    "humidity": "mean",
    "precipitation": "mean",
    "wind_speed": "mean",
    "BCCMASS": "mean",
    "CH4_column_volume_mixing_ratio_dry_air_bias_corrected": "mean",
    "CO_column_number_density": "mean",
    "DUCMASS": "mean",
    "NO2_column_number_density": "mean",
    "O3_column_number_density": "mean",
    "SO2_column_number_density": "mean",
    "absorbing_aerosol_index": "mean",
    "tropospheric_HCHO_column_number_density": "mean",
    "Health_Risk_Index": "mean",
    "Urban_Heat_Index": "mean",
    "Real_Estate_Risk": "mean",
    "Green_Score": "mean"
}).reset_index()


# **Step 5: Save Zone-Wise Data in a Single CSV**
zone_data.to_csv("sklm_zone_wise_analysis.csv", index=False)

# **Step 6: Visualization**
plt.figure(figsize=(10, 5))
sns.barplot(x="zone", y="Health_Risk_Index", data=zone_data, palette="Reds")
plt.title("Health Risk Index by Zone")
plt.xlabel("Zone")
plt.ylabel("Health Risk Index")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="zone", y="Urban_Heat_Index", data=zone_data, palette="Oranges")
plt.title("Urban Heat Index by Zone")
plt.xlabel("Zone")
plt.ylabel("Urban Heat Index")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="zone", y="Real_Estate_Risk", data=zone_data, palette="Purples")
plt.title("Real Estate Risk by Zone")
plt.xlabel("Zone")
plt.ylabel("Real Estate Risk Index")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="zone", y="Green_Score", data=zone_data, palette="Greens")
plt.title("Green Score by Zone")
plt.xlabel("Zone")
plt.ylabel("Green Score")
plt.show()