import pandas as pd
import numpy as np

# Load the CSV data
df = pd.read_csv("sklm_zone_wise_analysis.csv")

# # List of features to normalize
# features_to_normalize = [
#     'temperature', 'precipitation', 'wind_speed', 'absorbing_aerosol_index',
#     'bare', 'built', 'crops', 'trees', 'water',
#     'Health_Risk_Index', 'Urban_Heat_Index', 'Real_Estate_Risk', 'Green_Score'
# ]

# # Function for min-max normalization
# def normalize(series):
#     return (series - series.min()) / (series.max() - series.min())

# # Normalize each selected feature
# for feature in features_to_normalize:
#     norm_feature = feature + '_norm'
#     df[norm_feature] = normalize(df[feature])

# Step 2: Compute Hazard Score (H)
# Adjust weights as needed
w_temp = 0.3
w_precip = 0.3
w_wind = 0.2
w_aerosol = 0.2
df['H'] = (w_temp * df['temperature'] +
           w_precip * df['precipitation'] +
           w_wind * df['wind_speed'] +
           w_aerosol * df['absorbing_aerosol_index'])

# Step 3: Compute Exposure Score (E)
w_built = 0.4
w_crops = 0.2
w_bare = 0.2
w_trees = 0.1
w_water = 0.1
df['E'] = (w_built * df['built'] +
           w_crops * df['crops'] +
           w_bare * df['bare'] +
           w_trees * df['trees'] +
           w_water * df['water'])

# Step 4: Compute Vulnerability Score (V)
# Here, a higher Green_Score reduces vulnerability so it's subtracted.
alpha = 0.4
beta = 0.3
gamma = 0.2
delta = 0.1
df['V'] = (alpha * df['Health_Risk_Index'] +
           beta * df['Urban_Heat_Index'] +
           gamma * df['Real_Estate_Risk'] -
           delta * df['Green_Score'])

# Step 5: Compute Composite Climate Risk Index (CRI)
df['CRI'] = df['H'] * df['E'] * df['V']

# Optional: Categorize risk into quantiles (Low, Moderate, High, Very High)
df['Risk_Category'] = pd.qcut(df['CRI'], q=4, labels=['Low', 'Moderate', 'High', 'Very High'])
print(df['CRI'])
print(df['Risk_Category'])
# Save the updated dataframe to a new CSV file
df.to_csv("updated_data_with_cri_sklm.csv", index=False)

print("Climate Risk Index computation completed and saved to updated_data_with_cri.csv")
