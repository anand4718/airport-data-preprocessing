# AI Pioneers Internship - Week 1: Python for Machine Learning & Data Preprocessing
**Domain:** Airport Operations & Aviation Analytics  
**Program:** AI Pioneers Internship (Yuva Intern / NSDC)  
**Deliverable:** Sample Dataset Cleaning, EDA, Feature Engineering & ML Preprocessing Pipeline

---

## 📌 Project Overview
This repository contains the end-to-end data science pipeline developed for **Week 1** of the AI Pioneers Internship. In aviation logistics and airport management, operational datasets contain noise, missing sensor recordings, skewed flight delays, and multi-class categorical variables.

This project implements a modular, production-ready Python pipeline using **Pandas**, **NumPy**, **Scikit-Learn**, **Matplotlib**, and **Seaborn** to transform raw airport operational records into clean, standardized feature matrices ready for machine learning models.

---

## 📂 Repository Structure
```text
├── airport_operations_dataset.csv               # Raw airport operations dataset (500 records)
├── cleaned_airport_dataset.csv                 # Cleaned & ML-ready preprocessed dataset
├── airport_preprocessing.py                     # Main Python preprocessing pipeline script
├── Week_1_Airport_Data_Preprocessing_Report.docx # Comprehensive Word report for submission
├── plots/                                       # Exploratory Data Analysis (EDA) visualizations
│   ├── eda_missing_data.png                     # Missing value diagnosis chart
│   ├── eda_weather_delays.png                   # Delay distribution across weather conditions
│   ├── eda_correlation_heatmap.png              # Pearson correlation matrix
│   └── eda_scaling_comparison.png               # Preprocessing distribution before vs after scaling
└── README.md                                    # Project documentation
```

---

## 🛠️ Data Preprocessing Pipeline Steps

1. **Data Ingestion & Sanity Checks:** Ingested tabular flight records across major hubs (DEL, BOM, BLR, HYD, MAA, CCU) and inspected structure via `.info()` and `.describe()`.
2. **Exploratory Data Analysis (EDA):** Evaluated feature distributions, null-value frequencies, and correlations between cruising distance, flight duration, and departure delays.
3. **Missing Value Imputation:**
   - Continuous skewed metrics (`Flight_Duration_min`, `Distance_km`, `Passenger_Count`): Imputed with **Median**.
   - Turnaround metrics (`Baggage_Handling_min`): Imputed with **Mean**.
   - Categorical attributes (`Weather_Condition`): Imputed with **Mode**.
4. **Outlier Mitigation:** Applied Interquartile Range (**IQR**) capping on `Departure_Delay_min` to suppress extreme delay distortions without data loss.
5. **Feature Engineering:**
   - Synthesized `Calculated_Speed_kmh` (derived flight velocity).
   - Created binary classification target `Is_Delayed_30Min` (> 30 min threshold).
6. **Categorical Encoding:** Applied **One-Hot Encoding** (`pd.get_dummies(drop_first=True)`) to transform nominal variables (`Airline`, `Weather_Condition`, `Aircraft_Type`) into numeric dummy indicators.
7. **Feature Normalization / Standardization:** Scaled continuous metrics using **StandardScaler** ($z = \frac{x - \mu}{\sigma}$) to establish zero mean and unit variance.

---

## 🚀 How to Run the Pipeline

### Prerequisites
Install the required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn python-docx
```

### Execution
Run the preprocessing script:
```bash
python airport_preprocessing.py
```

---

## 📊 Summary of Pipeline Results

| Metric | Raw Dataset (Before) | Cleaned Dataset (After) |
| :--- | :--- | :--- |
| **Total Missing Values** | 95 missing entries | **0 (100% Imputed)** |
| **Feature Dimensions** | 500 rows × 12 columns | **500 rows × 21 columns** |
| **Categorical Format** | Text strings (IndiGo, Fog, etc.) | **One-Hot Encoded (0/1)** |
| **Feature Scaling** | Unstandardized (10 – 2500+) | **Standardized ($z \approx 0.0, \sigma \approx 1.0$)** |

---

## 👨‍💻 Author & Submission
- **Internship:** AI Pioneers Internship (Yuva Intern / NSDC)
- **Topic:** Airport Operational Analytics & Machine Learning Preprocessing
