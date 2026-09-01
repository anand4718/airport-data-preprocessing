"""
AI Pioneers Internship - Week 1: Python for Machine Learning & Data Preprocessing
Domain: Airport Operations & Flight Delay Analytics
Deliverable: End-to-end data cleaning, EDA, feature engineering, categorical encoding, and scaling pipeline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(filepath="airport_operations_dataset.csv"):
    """Loads dataset and prints structural summary."""
    print("="*60)
    print("STEP 1: LOADING AIRPORT DATASET")
    print("="*60)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found. Please verify filepath.")
    
    df = pd.read_csv(filepath)
    print(f"Dataset Loaded Successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nDataset Info:")
    df.info()
    return df

def perform_eda(df):
    """Executes initial Exploratory Data Analysis."""
    print("\n" + "="*60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    print("\n1. Statistical Summary (Numerical Features):")
    print(df.describe().round(2))
    
    print("\n2. Missing Value Analysis:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print("\n3. Categorical Distributions:")
    print("Airlines breakdown:\n", df['Airline'].value_counts())
    print("\nWeather conditions:\n", df['Weather_Condition'].value_counts())

def handle_missing_values(df):
    """Imputes missing numerical and categorical values."""
    print("\n" + "="*60)
    print("STEP 3: HANDLING MISSING VALUES")
    print("="*60)
    df_clean = df.copy()
    
    # Numerical imputation via Median (robust against skew)
    num_impute_cols = ['Flight_Duration_min', 'Distance_km', 'Passenger_Count']
    for col in num_impute_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f"Imputed missing '{col}' with median: {median_val:.2f}")
        
    # Gaussian feature imputation via Mean
    mean_baggage = df_clean['Baggage_Handling_min'].mean()
    df_clean['Baggage_Handling_min'] = df_clean['Baggage_Handling_min'].fillna(mean_baggage)
    print(f"Imputed missing 'Baggage_Handling_min' with mean: {mean_baggage:.2f}")

    # Categorical imputation via Mode
    mode_weather = df_clean['Weather_Condition'].mode()[0]
    df_clean['Weather_Condition'] = df_clean['Weather_Condition'].fillna(mode_weather)
    print(f"Imputed missing 'Weather_Condition' with mode: '{mode_weather}'")

    print(f"Remaining null values: {df_clean.isnull().sum().sum()}")
    return df_clean

def handle_outliers_iqr(df, col='Departure_Delay_min'):
    """Identifies and caps severe outliers using Interquartile Range (IQR)."""
    print("\n" + "="*60)
    print(f"STEP 4: OUTLIER DETECTION & TREATMENT ({col})")
    print("="*60)
    df_out = df.copy()
    q1 = df_out[col].quantile(0.25)
    q3 = df_out[col].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + (1.5 * iqr)
    lower_limit = max(0, q1 - (1.5 * iqr))
    
    outlier_count = (df_out[col] > upper_limit).sum()
    print(f"Q1: {q1:.2f} min | Q3: {q3:.2f} min | IQR: {iqr:.2f} min")
    print(f"Upper Bound: {upper_limit:.2f} min | Outliers Identified: {outlier_count}")
    
    # Cap upper outliers
    df_out[f"{col}_capped"] = np.where(df_out[col] > upper_limit, upper_limit, df_out[col])
    print(f"Outliers successfully capped at {upper_limit:.2f} minutes.")
    return df_out

def feature_engineering(df):
    """Derives domain-specific features and predictive targets."""
    print("\n" + "="*60)
    print("STEP 5: FEATURE ENGINEERING & DERIVATION")
    print("="*60)
    df_fe = df.copy()
    
    # 1. Calculated flight speed (km/h)
    df_fe['Calculated_Speed_kmh'] = (df_fe['Distance_km'] / (df_fe['Flight_Duration_min'] / 60)).round(1)
    
    # 2. Binary target: Severe Flight Delay (> 30 minutes)
    df_fe['Is_Delayed_30Min'] = (df_fe['Departure_Delay_min'] > 30).astype(int)
    
    print("Engineered Features:")
    print("• 'Calculated_Speed_kmh' -> Distance / Duration")
    print("• 'Is_Delayed_30Min'     -> Classification Target (1 = Delay > 30 min, 0 = On-Time/Minor)")
    return df_fe

def encode_and_scale(df):
    """Applies One-Hot Encoding and Feature Standardization."""
    print("\n" + "="*60)
    print("STEP 6: CATEGORICAL ENCODING & FEATURE SCALING")
    print("="*60)
    df_proc = df.copy()
    
    # Categorical One-Hot Encoding (drop_first to avoid dummy variable trap)
    cat_cols = ['Airline', 'Weather_Condition', 'Aircraft_Type']
    encoded_dummies = pd.get_dummies(df_proc[cat_cols], drop_first=True, dtype=int)
    df_proc = pd.concat([df_proc, encoded_dummies], axis=1)
    print(f"Encoded categorical features: {cat_cols} -> Generated {encoded_dummies.shape[1]} dummy columns")

    # Feature Scaling with StandardScaler
    scaler = StandardScaler()
    num_scale_cols = ['Distance_km', 'Flight_Duration_min', 'Passenger_Count', 'Baggage_Handling_min', 'Calculated_Speed_kmh']
    scaled_matrix = scaler.fit_transform(df_proc[num_scale_cols])
    
    for idx, col in enumerate(num_scale_cols):
        df_proc[f"{col}_scaled"] = scaled_matrix[:, idx].round(4)
    print(f"Standardized numerical columns: {num_scale_cols}")
    
    # Feature Selection: Drop raw ID columns
    features_to_drop = ['Flight_ID', 'Runway_Used', 'Origin_Airport', 'Destination_Airport']
    df_final = df_proc.drop(columns=features_to_drop)
    print(f"Dropped high-cardinality identifiers: {features_to_drop}")
    
    return df_final

def main():
    print("="*60)
    print("AIRPORT OPERATIONS ML PREPROCESSING PIPELINE")
    print("AI Pioneers Internship - NSDC / Yuva Intern")
    print("="*60)
    
    # Pipeline execution
    df_raw = load_data("airport_operations_dataset.csv")
    perform_eda(df_raw)
    df_imputed = handle_missing_values(df_raw)
    df_outlier = handle_outliers_iqr(df_imputed)
    df_fe = feature_engineering(df_outlier)
    df_final = encode_and_scale(df_fe)
    
    # Save cleaned output
    output_path = "cleaned_airport_dataset.csv"
    df_final.to_csv(output_path, index=False)
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED")
    print(f"Processed dataset saved to: {output_path}")
    print(f"Final Data Shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print("="*60)

if __name__ == "__main__":
    main()
