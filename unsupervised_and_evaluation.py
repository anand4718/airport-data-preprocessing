"""
AI Pioneers Internship - Week 3: Unsupervised Learning & Model Evaluation
Domain: Airport Operations & Aviation Analytics
Deliverables: Implementation of K-Means Clustering, Hierarchical Agglomerative Clustering,
             PCA Dimensionality Reduction, 5-Fold Cross-Validation, and GridSearchCV Hyperparameter Tuning.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage

def load_data(filepath="cleaned_airport_dataset.csv"):
    """Loads cleaned airport operational dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found. Run preprocessing first.")
    df = pd.read_csv(filepath)
    print(f"[OK] Cleaned Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def prepare_data(df):
    """Prepares feature matrix X and targets."""
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
    y_clf = df['Is_Delayed_30Min']
    return X, y_clf

def run_pca_and_kmeans(df, X):
    """Executes PCA dimensionality reduction and K-Means clustering."""
    print("\n" + "="*60)
    print("1. DIMENSIONALITY REDUCTION (PCA) & K-MEANS CLUSTERING")
    print("="*60)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA Variance Explained: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")

    # Optimal K-Means (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X)
    sil = silhouette_score(X, df['KMeans_Cluster'])
    print(f"K-Means (k=3) Silhouette Score: {sil:.4f}")

    # Cluster Profiles
    profiles = df.groupby('KMeans_Cluster').agg({
        'Distance_km': 'mean',
        'Flight_Duration_min': 'mean',
        'Departure_Delay_min': 'mean',
        'Is_Delayed_30Min': 'mean'
    }).round(2)
    profiles['Flight_Count'] = df['KMeans_Cluster'].value_counts()
    print("\nCluster Operational Profiles:")
    print(profiles)
    return df, kmeans, pca

def run_hierarchical_clustering(df, X):
    """Runs Hierarchical Agglomerative Clustering."""
    print("\n" + "="*60)
    print("2. HIERARCHICAL AGGLOMERATIVE CLUSTERING")
    print("="*60)
    agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
    df['Hierarchical_Cluster'] = agg.fit_predict(X)
    ari = adjusted_rand_score(df['KMeans_Cluster'], df['Hierarchical_Cluster'])
    print(f"Hierarchical Clustering fitted. Adjusted Rand Index with K-Means: {ari:.4f}")

def run_kfold_cross_validation(X, y):
    """Performs Stratified 5-Fold Cross-Validation."""
    print("\n" + "="*60)
    print("3. STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("="*60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    for name, model in models.items():
        acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
        auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        results.append({
            'Model': name,
            'Mean CV Accuracy (%)': round(acc.mean() * 100, 2),
            'Std (±%)': round(acc.std() * 100, 2),
            'Mean F1 (%)': round(f1.mean() * 100, 2),
            'Mean ROC-AUC': round(auc.mean(), 4)
        })
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df

def run_hyperparameter_tuning(X, y):
    """Conducts GridSearchCV on Random Forest."""
    print("\n" + "="*60)
    print("4. HYPERPARAMETER TUNING VIA GRIDSEARCHCV")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 8],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Optimal Hyperparameters: {grid.best_params_}")
    
    tuned_preds = grid.best_estimator_.predict(X_test)
    tuned_probs = grid.best_estimator_.predict_proba(X_test)[:, 1]
    
    print(f"Tuned Model Test Accuracy: {accuracy_score(y_test, tuned_preds)*100:.2f}%")
    print(f"Tuned Model Test Precision: {precision_score(y_test, tuned_preds)*100:.2f}%")
    print(f"Tuned Model Test F1-Score: {f1_score(y_test, tuned_preds)*100:.2f}%")
    print(f"Tuned Model Test ROC-AUC: {roc_auc_score(y_test, tuned_probs):.4f}")

def main():
    print("="*60)
    print("AI PIONEERS INTERNSHIP - WEEK 3 UNSUPERVISED & EVALUATION PIPELINE")
    print("="*60)
    df = load_data("cleaned_airport_dataset.csv")
    X, y_clf = prepare_data(df)
    df, km, pca = run_pca_and_kmeans(df, X)
    run_hierarchical_clustering(df, X)
    run_kfold_cross_validation(X, y_clf)
    run_hyperparameter_tuning(X, y_clf)
    print("\n" + "="*60)
    print("WEEK 3 PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
