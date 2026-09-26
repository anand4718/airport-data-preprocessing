# ✈️ AeroPredict: Airport Operations Flight Delay Prediction & Management System
**AI Pioneers Internship (Yuva Intern / NSDC) — Final Capstone Project**  
**Author:** Anand Patgar  
**Domain:** Aviation Logistics, Real-Time Airport Operations & Applied Machine Learning  

---

## 📌 Executive Summary
**AeroPredict** is an end-to-end production Machine Learning application designed to forecast and mitigate commercial flight delays. Built across a 4-week rigorous curriculum under the **AI Pioneers Internship**, the project spans the complete AI lifecycle: from raw tabular data ingestion and statistical preprocessing to supervised delay modeling, unsupervised operational clustering, model serialization, and microservice deployment via a **Flask REST API** and interactive **Web Dashboard**.

---

## 📂 Repository Structure
```text
├── airport_operations_dataset.csv               # Raw airport operations dataset (500 records)
├── cleaned_airport_dataset.csv                 # Cleaned & preprocessed ML-ready dataset
├── airport_preprocessing.py                     # Week 1: Data Cleaning, Imputation & Scaling
├── supervised_learning.py                       # Week 2: Supervised ML (Regression & Classification)
├── unsupervised_and_evaluation.py               # Week 3: Clustering, PCA & 5-Fold Cross-Validation
├── app.py                                       # Week 4: Flask REST API & Web Server
├── test_api.py                                  # Week 4: Automated Model Inference Test Suite
├── templates/
│   └── index.html                               # Modern HTML5/CSS3 Dispatcher Dashboard UI
├── models/
│   ├── airport_delay_classifier.joblib          # Serialized Random Forest Classifier
│   ├── airport_delay_regressor.joblib           # Serialized Random Forest Regressor
│   ├── scaler.joblib                            # Serialized StandardScaler
│   └── model_metadata.json                      # Model versions, schemas, and metrics
├── Week_1_Airport_Data_Preprocessing_Report.docx # Week 1 Milestone Report (.docx)
├── Week_2_Supervised_Learning_Report.docx       # Week 2 Milestone Report (.docx)
├── Week_3_Unsupervised_Learning_Report.docx     # Week 3 Milestone Report (.docx)
├── Week_4_AI_Project_Deployment_Report.docx     # Week 4 Final Capstone Report (.docx)
├── plots/                                       # Analytical Visualizations & Evaluation Charts
│   ├── eda_missing_data.png                     # Missing value diagnostics
│   ├── eda_weather_delays.png                   # Delay distribution across weather
│   ├── eda_correlation_heatmap.png              # Pearson correlation matrix
│   ├── eda_scaling_comparison.png               # StandardScaler distribution comparison
│   ├── model_regression_comparison.png          # MAE, RMSE, R² comparison of Regressors
│   ├── model_classification_comparison.png      # Accuracy, Precision, Recall, F1 comparison
│   ├── classification_confusion_matrices.png    # 2x2 grid of confusion matrices
│   ├── roc_curves_comparison.png                # ROC-AUC curves for all classifiers
│   ├── feature_importance_rf.png                # Top 10 Random Forest feature importances
│   ├── kmeans_elbow_silhouette.png              # Elbow method and Silhouette coefficient curves
│   ├── pca_clusters_visualization.png           # 2D PCA projection of flight operational clusters
│   ├── hierarchical_dendrogram.png              # Hierarchical Ward linkage dendrogram
│   ├── kfold_cross_validation_comparison.png    # 5-Fold Stratified CV benchmark comparison
│   ├── hyperparameter_tuning_comparison.png    # Baseline vs Tuned Random Forest comparison
│   └── system_architecture.png                  # End-to-end system deployment architecture
└── README.md                                    # Complete project documentation
```

---

## 🗓️ Complete 4-Week Technical Progression

### 🔹 Week 1: Python for ML & Data Preprocessing
- **Data Ingestion:** Ingested 500 commercial flight records across major Indian hubs (DEL, BOM, BLR, HYD, MAA, CCU).
- **Statistical Imputation:** Resolved missing values via Median (continuous skewed features), Mean (Gaussian baggage turnaround), and Mode (weather conditions).
- **Outlier Capping:** Applied Interquartile Range (**IQR**) filtering to cap extreme delay spikes without row deletion.
- **Feature Engineering & Scaling:** Derived `Calculated_Speed_kmh`, one-hot encoded nominal attributes, and standardized continuous columns with `StandardScaler` ($z = \frac{x - \mu}{\sigma}$).

### 🔹 Week 2: Supervised Machine Learning Models
- **Regression (Delay Duration in Minutes):** Evaluated Linear Regression, Decision Tree, Random Forest, and KNN. Decision Tree and Linear Regression achieved **MAE $\approx$ 14.0 mins** and **$R^2 = 0.43$**.
- **Classification (Severe Delay > 30 mins):** Evaluated Logistic Regression (**83.0% Accuracy, 92.0% Precision, 73.0% F1**), Random Forest (**82.0% Accuracy**), Decision Tree (**79.0%**), and KNN (**64.0%**).

### 🔹 Week 3: Unsupervised Learning & Advanced Evaluation
- **Dimensionality Reduction:** Computed **Principal Component Analysis (PCA)** to project 17 features onto 2 orthogonal principal components.
- **Clustering:** Determined optimal $k = 3$ clusters using the **Elbow Method (Inertia)** and **Silhouette Score**, identifying Regional Connectors, Heavy Long-Haul, and Weather-Disrupted flight corridors.
- **Hierarchical Dendrogram:** Validated cluster boundaries using Ward's minimum variance agglomeration.
- **Rigorous Evaluation:** Conducted **5-Fold Stratified Cross-Validation** (Logistic Regression: **81.6% ± 1.6%**) and **GridSearchCV** hyperparameter tuning.

### 🔹 Week 4: Model Deployment & Final Capstone
- **Model Serialization:** Serialized dual-head ML estimators and standard scalers using **Joblib** into the `models/` registry.
- **Flask REST API:** Built production endpoints:
  - `GET /`: Interactive web dashboard for operations dispatchers.
  - `GET /health`: Microservice telemetry and health status.
  - `POST /api/predict`: JSON inference API returning delay probability, estimated delay minutes, risk tiers (Low/Medium/High), and actionable ground staff recommendations.
- **Automated Verification:** Verified sub-50ms inference latency across multiple flight operational test cases (`test_api.py`).

---

## 🚀 Quick Start & Deployment Guide

### 1. Install Dependencies
```bash
pip install flask joblib scikit-learn pandas numpy matplotlib seaborn scipy python-docx
```

### 2. Launch the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

### 3. Run Automated Model Verification
```bash
python test_api.py
```

### 4. Sample API Request via cURL
```bash
curl -X POST http://127.0.0.1:5000/api/predict      -H "Content-Type: application/json"      -d '{
       "airline": "SpiceJet",
       "aircraft_type": "Boeing 737 MAX",
       "weather_condition": "Thunderstorm",
       "distance_km": 800.0,
       "flight_duration_min": 110.0,
       "passenger_count": 185,
       "baggage_handling_min": 38.0
     }'
```

**Sample Response:**
```json
{
  "status": "success",
  "is_delayed": true,
  "delay_probability": 89.5,
  "estimated_delay_minutes": 76.6,
  "risk_level": "High",
  "prediction_summary": "⚠️ High Probability of Severe Flight Delay (>30 mins)",
  "recommendation": "Adverse conditions detected (Weather/Turnaround). Issue ground hold warning, notify passengers, and prioritize apron turnaround sequence."
}
```

---

## 👨‍💻 Author & Submission
- **Internship:** AI Pioneers Internship (Yuva Intern / NSDC)
- **Author:** Anand Patgar
- **Repository:** [https://github.com/anand4718/airport-data-preprocessing](https://github.com/anand4718/airport-data-preprocessing)
