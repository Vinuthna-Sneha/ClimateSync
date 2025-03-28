import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
file_path = "srikakulam_full_dataset_with_zones.csv"  # Update with actual path
df = pd.read_csv(file_path)

# Drop missing values
df.dropna(inplace=True)

# Normalize Features
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

# Clustering into 12 Zones
num_clusters = 12
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['zone'] = kmeans.fit_predict(df[['latitude', 'longitude']])

# Compute Impact Indices
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

df["Climate_Risk_Index"] = (
    df["temperature"] * 0.3 +
    df["precipitation"] * 0.3 +
    df["humidity"] * 0.2 +
    df["wind_speed"] * 0.2
)

# Compute Hazard Score (H), Exposure Score (E), and Vulnerability Score (V)
df['Hazard_Score'] = (0.3 * df['temperature'] +
           0.3 * df['precipitation'] +
           0.2 * df['wind_speed'] +
           0.2 * df['absorbing_aerosol_index'])

df['Exposure_Score'] = (0.4 * df['built'] +
           0.2 * df['crops'] +
           0.2 * df['bare'] +
           0.1 * df['trees'] +
           0.1 * df['water'])

df['Vulnerability_Index'] = (0.4 * df['Health_Risk_Index'] +
           0.3 * df['Urban_Heat_Index'] +
           0.2 * df['Real_Estate_Risk'] -
           0.1 * df['Green_Score'])

df['Climate_Risk_Index'] = df['Hazard_Score'] * df['Exposure_Score'] * df['Vulnerability_Index']

# Prepare for SHAP Analysis
features = features_to_scale
indices = ["Hazard_Score", "Exposure_Score", "Vulnerability_Index", "Health_Risk_Index", "Urban_Heat_Index", "Real_Estate_Risk", "Green_Score", "Climate_Risk_Index", "Climate_Risk_Index"]

zone_trends = {}

# Loop through each zone and analyze trends
for zone in df["zone"].unique():
    zone_df = df[df["zone"] == zone].sample(n=min(500, len(df[df["zone"] == zone])), random_state=42)

    zone_trends[zone] = {}

    for index in indices:
        X = zone_df[features]
        y = zone_df[index]

        if X.shape[0] < 10:  # Ensure there is enough data
            continue

        # Train a simpler, faster model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        # Faster SHAP method
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X, check_additivity=False)

        mean_shap = np.abs(shap_values.values).mean(axis=0)
        top_feature_idx = np.argmax(mean_shap)
        top_feature = features[top_feature_idx]
        top_feature_impact = mean_shap[top_feature_idx]

        direction = "increases" if np.corrcoef(X[top_feature], y)[0, 1] > 0 else "decreases"

        zone_trends[zone][index] = {
            "feature": top_feature,
            "impact": round(top_feature_impact, 3),
            "effect": direction
        }

# Print results
for zone, trends in zone_trends.items():
    print(f"**Zone {zone} Trends:**")
    for index, details in trends.items():
        print(f"  - {index} is primarily influenced by **{details['feature']}** (impact: {details['impact']}) and {details['effect']} as it changes.")
    print()

    import pandas as pd

    # Assuming 'zone_trends' is already defined as in your provided code.
    
    # Create a list to store the data for the CSV
    csv_data = []
    
    # Iterate through the zone_trends dictionary
    for zone, trends in zone_trends.items():
        for index, details in trends.items():
            csv_data.append([zone, index, details['feature'], details['impact'], details['effect']])
    
    # Create a Pandas DataFrame from the data
    df_zone_trends = pd.DataFrame(csv_data, columns=['Zone', 'Trend_Description', 'Influencing_Feature', 'Impact', 'Effect'])
    
    # Save the DataFrame to a CSV file
    df_zone_trends.to_csv('shap_insights_sklm.csv', index=False)