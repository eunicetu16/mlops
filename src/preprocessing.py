import pandas as pd
import os

# Create the output directory if needed
os.makedirs("data", exist_ok=True)

# Load the used car dataset
input_path = "data/cars24data.csv"  # make sure this file is added with DVC
output_path = "data/processed_data.csv"

used_cars = pd.read_csv(input_path)


# Encode categorical variables
categorical_cols = ['Spare key', 'Transmission', 'Fuel type']
used_cars_encoded = pd.get_dummies(used_cars, columns=categorical_cols, drop_first=True)

# Optional: drop columns not used for modeling
if 'Model Name' in used_cars_encoded.columns:
    used_cars_encoded.drop(columns=['Model Name'], inplace=True)

# Save processed data
used_cars_encoded.to_csv(output_path, index=False)

