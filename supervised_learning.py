"""
AI Pioneers Internship - Week 2: Supervised Machine Learning Models
Domain: Airport Operations & Flight Delay Analytics
Deliverables: Implementation of Linear Regression, Logistic Regression, Decision Trees,
             Random Forest, and K-Nearest Neighbors (KNN) with comprehensive evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)

def load_data(filepath="cleaned_airport_dataset.csv"):
    """Loads the preprocessed airport operational dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found. Run Week 1 preprocessing first.")
    df = pd.read_csv(filepath)
    print(f"[OK] Cleaned Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def prepare_features_and_targets(df):
    """Prepares feature matrix X and targets y_reg and y_clf."""
    feature_cols = [
        'Distance_km_scaled', 'Flight_Duration_min_scaled', 'Passenger_Count_scaled',
        'Baggage_Handling_min_scaled', 'Calculated_Speed_kmh_scaled',
        'Airline_Akasa Air', 'Airline_IndiGo', 'Airline_SpiceJet', 'Airline_Vistara',
        'Weather_Condition_Fog', 'Weather_Condition_Overcast', 'Weather_Condition_Rain',
        'Weather_Condition_Thunderstorm', 'Aircraft_Type_Airbus A320neo',
        'Aircraft_Type_Airbus A321neo', 'Aircraft_Type_Boeing 737 MAX',
        'Aircraft_Type_Boeing 787 Dreamliner'
    ]
    
    X = df[feature_cols]
    y_reg = df['Departure_Delay_min_capped']
    y_clf = df['Is_Delayed_30Min']
    return X, y_reg, y_clf, feature_cols

def train_and_evaluate_regression(X, y_reg):
    """Trains and compares Linear Regression, Decision Tree, Random Forest, and KNN Regressors."""
    print("\n" + "="*60)
    print("1. REGRESSION MODELS: PREDICTING FLIGHT DELAY DURATION (MINUTES)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        'KNN Regressor (k=5)': KNeighborsRegressor(n_neighbors=5)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        results.append({
            'Model': name,
            'MAE (min)': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE (min)': round(rmse, 2),
            'R2 Score': round(r2, 4)
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df

def train_and_evaluate_classification(X, y_clf):
    """Trains and compares Logistic Regression, Decision Tree, Random Forest, and KNN Classifiers."""
    print("\n" + "="*60)
    print("2. CLASSIFICATION MODELS: PREDICTING CRITICAL DELAY (> 30 MIN)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree Classifier': DecisionTreeClassifier(max_depth=4, random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'KNN Classifier (k=5)': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        
        results.append({
            'Model': name,
            'Accuracy (%)': round(acc * 100, 2),
            'Precision (%)': round(prec * 100, 2),
            'Recall (%)': round(rec * 100, 2),
            'F1-Score (%)': round(f1 * 100, 2),
            'ROC-AUC': round(auc, 4)
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df

def main():
    print("="*60)
    print("AI PIONEERS INTERNSHIP - WEEK 2 SUPERVISED LEARNING PIPELINE")
    print("="*60)
    df = load_data("cleaned_airport_dataset.csv")
    X, y_reg, y_clf, _ = prepare_features_and_targets(df)
    train_and_evaluate_regression(X, y_reg)
    train_and_evaluate_classification(X, y_clf)
    print("\n" + "="*60)
    print("SUPERVISED LEARNING TRAINING & EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
