# AI Pioneers Internship - Airport Operations & Flight Delay Analytics
**Program:** AI Pioneers Internship (Yuva Intern / NSDC)  
**Domain:** Aviation Operations, Flight Logistics & Predictive Machine Learning  

---

## 📂 Repository Structure
```text
├── airport_operations_dataset.csv               # Raw airport operations dataset (500 records)
├── cleaned_airport_dataset.csv                 # Cleaned & ML-ready preprocessed dataset
├── airport_preprocessing.py                     # Week 1: Data Preprocessing & EDA Pipeline
├── supervised_learning.py                       # Week 2: Supervised Learning Models (Reg & Clf)
├── Week_1_Airport_Data_Preprocessing_Report.docx # Week 1 Submission Report (.docx)
├── Week_2_Supervised_Learning_Report.docx       # Week 2 Submission Report (.docx)
├── plots/                                       # Visualizations & Evaluation Charts
│   ├── eda_missing_data.png                     # Missing value diagnostics
│   ├── eda_weather_delays.png                   # Delay distribution across weather
│   ├── eda_correlation_heatmap.png              # Pearson correlation matrix
│   ├── eda_scaling_comparison.png               # StandardScaler distribution comparison
│   ├── model_regression_comparison.png          # MAE, RMSE, R² comparison of Regressors
│   ├── model_classification_comparison.png      # Accuracy, Precision, Recall, F1 comparison
│   ├── classification_confusion_matrices.png    # 2x2 grid of confusion matrices
│   ├── roc_curves_comparison.png                # ROC-AUC curves for all classifiers
│   └── feature_importance_rf.png                # Top 10 Random Forest feature importances
└── README.md                                    # Comprehensive project documentation
```

---

## 🚀 Week 1: Data Preprocessing & EDA Pipeline
- **Dataset:** 500 commercial flight records across major Indian hubs (DEL, BOM, BLR, HYD, MAA, CCU).
- **Imputation:** Median for continuous skewed variables, Mean for Gaussian baggage turnaround, Mode for categorical weather attributes.
- **Outlier Treatment:** Interquartile Range (**IQR**) capping for extreme departure delays.
- **Feature Engineering:** Derived `Calculated_Speed_kmh` and binary target `Is_Delayed_30Min`.
- **Encoding & Normalization:** One-Hot Encoding (`drop_first=True`) and `StandardScaler` ($z = \frac{x - \mu}{\sigma}$).

---

## 🤖 Week 2: Supervised Machine Learning Models

### 1. Regression Models (Predicting Delay in Minutes)
Target: Continuous departure delay (`Departure_Delay_min_capped`)

| Model Architecture | MAE (min) | MSE | RMSE (min) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree Regressor** | **13.96** | **341.54** | **18.48** | **0.4298** |
| **Linear Regression** | 14.02 | 343.63 | 18.54 | 0.4263 |
| **Random Forest Regressor** | 14.09 | 347.98 | 18.65 | 0.4190 |
| **KNN Regressor ($k=5$)** | 19.80 | 640.92 | 25.32 | -0.0700 |

### 2. Classification Models (Predicting Delay > 30 mins)
Target: Binary indicator (`Is_Delayed_30Min`)

| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **83.00%** | **92.00%** | **60.53%** | **73.02%** | **0.7857** |
| **Random Forest Classifier** | 82.00% | 91.67% | 57.89% | 70.97% | 0.7330 |
| **Decision Tree Classifier** | 79.00% | 90.48% | 50.00% | 64.41% | 0.7912 |
| **KNN Classifier ($k=5$)** | 64.00% | 55.00% | 28.95% | 37.93% | 0.6352 |

---

## 🛠️ How to Execute

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn python-docx
```

### 2. Run Preprocessing Pipeline (Week 1)
```bash
python airport_preprocessing.py
```

### 3. Run Supervised Machine Learning Pipeline (Week 2)
```bash
python supervised_learning.py
```

---

## 👨‍💻 Author & Submission
- **Internship:** AI Pioneers Internship (Yuva Intern / NSDC)
- **Repository:** [airport-data-preprocessing](https://github.com/anand4718/airport-data-preprocessing)
