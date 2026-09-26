# AI Pioneers Internship - Airport Operations & Flight Analytics Pipeline
**Program:** AI Pioneers Internship (Yuva Intern / NSDC)  
**Domain:** Aviation Logistics, Flight Delay Analytics & Applied Machine Learning  

---

## 📂 Repository Structure
```text
├── airport_operations_dataset.csv               # Raw airport operations dataset (500 records)
├── cleaned_airport_dataset.csv                 # Cleaned & preprocessed ML-ready dataset
├── airport_preprocessing.py                     # Week 1: Data Cleaning, Imputation & Scaling Pipeline
├── supervised_learning.py                       # Week 2: Supervised Learning (Regression & Classification)
├── unsupervised_and_evaluation.py               # Week 3: Unsupervised Clustering & Advanced Evaluation
├── Week_1_Airport_Data_Preprocessing_Report.docx # Week 1 Submission Report (.docx)
├── Week_2_Supervised_Learning_Report.docx       # Week 2 Submission Report (.docx)
├── Week_3_Unsupervised_Learning_Report.docx     # Week 3 Submission Report (.docx)
├── plots/                                       # Visualizations & Evaluation Charts
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
│   └── hyperparameter_tuning_comparison.png    # Baseline vs Tuned Random Forest comparison
└── README.md                                    # Comprehensive project documentation
```

---

## 🚀 Overview of Weekly Milestones

### 🔹 Week 1: Python for ML & Data Preprocessing
- **Data Ingestion & Cleaning:** Handled missing flight durations, cruising distances, and weather conditions via median, mean, and mode imputation.
- **Outlier Capping:** Applied Interquartile Range (**IQR**) capping on departure delays.
- **Feature Engineering & Scaling:** Synthesized `Calculated_Speed_kmh`, one-hot encoded nominal categories, and standardized features via `StandardScaler`.

### 🔹 Week 2: Supervised Machine Learning Models
- **Regression (Delay Minutes):** Trained Linear Regression, Decision Tree, Random Forest, and KNN. Decision Tree & Linear Regression achieved **MAE of ~14.0 mins** and **$R^2 \approx 0.43$**.
- **Classification (Severe Delay > 30m):** Trained Logistic Regression (**83.0% Accuracy, 92.0% Precision**), Random Forest (**82.0% Accuracy**), Decision Tree (**79.0%**), and KNN (**64.0%**).

### 🔹 Week 3: Unsupervised Learning & Model Evaluation
- **Dimensionality Reduction:** Computed **Principal Component Analysis (PCA)** to extract principal axes of variance for 2D visualization.
- **K-Means Clustering:** Evaluated cluster range $k \in [2, 6]$ using the **Elbow Method (Inertia)** and **Silhouette Coefficient**, establishing $k = 3$ as optimal.
- **Hierarchical Clustering:** Built an agglomerative **Dendrogram** with Ward linkage, confirming 3 operational flight segments.
- **5-Fold Stratified Cross-Validation:** Validated model generalization across 5 folds (Logistic Regression: **81.6% ± 1.6%**, Random Forest: **81.4% ± 2.2%**).
- **Hyperparameter Optimization:** Executed **GridSearchCV** over 54 parameter combinations for Random Forest, achieving **91.67% test precision** and **0.753 ROC-AUC**.

---

## 📊 Summary of Week 3 Benchmark Results

### 1. K-Means Operational Flight Clusters ($k=3$)
| Cluster | Profile | Flight Count | Avg Distance | Avg Duration | Avg Delay | Severe Delay Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | Medium-Haul Regional | 154 | 1,220 km | 182 min | 28.3 min | 34.0% |
| **1** | Long-Haul Heavy Trunk | 172 | 1,423 km | 209 min | 27.5 min | 38.0% |
| **2** | Weather-Disrupted Short Hub | 174 | 714 km | 114 min | 31.6 min | **42.0%** |

### 2. 5-Fold Stratified Cross-Validation Benchmark
| Classifier | Mean CV Accuracy (%) | Standard Deviation (±%) | Mean CV F1-Score (%) | Mean CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **81.60%** | **±1.62%** | **70.18%** | **0.7676** |
| **Random Forest** | 81.40% | ±2.24% | 69.50% | 0.7274 |
| **Decision Tree** | 80.60% | ±1.36% | 68.09% | 0.7593 |
| **KNN ($k=5$)** | 62.20% | ±3.97% | 34.04% | 0.5877 |

---

## 🛠️ How to Execute

### 1. Setup Environment
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn python-docx
```

### 2. Run All Pipelines
```bash
# Week 1 Preprocessing
python airport_preprocessing.py

# Week 2 Supervised Learning
python supervised_learning.py

# Week 3 Unsupervised Learning & Advanced Evaluation
python unsupervised_and_evaluation.py
```

---

## 👨‍💻 Author & Repository Link
- **Internship:** AI Pioneers Internship (Yuva Intern / NSDC)
- **Repository:** [https://github.com/anand4718/airport-data-preprocessing](https://github.com/anand4718/airport-data-preprocessing)
