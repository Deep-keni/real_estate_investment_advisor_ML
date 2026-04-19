# real_estate_investment_advisor_ML
# 🏠 Real Estate Investment Advisor

A Machine Learning application that helps potential investors make smarter real estate decisions by predicting property profitability and future value.

---

## 📌 Project Overview

This project solves two key problems for real estate investors:

- **Classification** → Is this property a *Good Investment*? (Yes/No with confidence score)
- **Regression** → What will this property's price be *after 5 years*?

Built using Python, Scikit-learn, XGBoost, MLflow, and deployed via Streamlit.

---

## 🎯 Features

- Predict whether a property is a good investment based on multiple features
- Estimate property price after 5 years
- Interactive UI with dropdowns, sliders, and number inputs
- EDA insights with 10 visualizations
- MLflow experiment tracking for model comparison
- SMOTE applied to handle class imbalance

---

## 🗂️ Project Structure

```
Real-Estate-Investment-Advisor/
│
├── data/
│   ├── india_housing_prices.csv          # Raw dataset
│   └── processed_data/
│       └── cleaned_data.csv              # Cleaned & preprocessed data
│
├── models/
│   ├── classifier.pkl                    # Saved XGBoost Classifier
│   └── regressor.pkl                     # Saved XGBoost Regressor
│
├── visuals/                              # EDA plots (auto-generated)
│
├── mlruns/                               # MLflow experiment logs
│
├── data_preprocessing.py                 # Data cleaning, encoding, scaling, feature engineering
├── eda.py                                # Exploratory Data Analysis (10 plots)
├── model_train.py                        # Model training + MLflow tracking
├── predict.py                            # Prediction functions used by Streamlit
├── app.py                                # Streamlit web application
├── requirements.txt                      # All dependencies
└── README.md
```

---

## 📊 Dataset

- **Source:** `india_housing_prices.csv`
- **Size:** 250,000 rows
- **Domain:** Indian Real Estate
- **Key Features:** BHK, Size, Price, Location, Property Type, Amenities, Infrastructure scores and more

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| ML Models | XGBoost, Random Forest |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Experiment Tracking | MLflow |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/real-estate-investment-advisor.git
cd real-estate-investment-advisor
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Data Preprocessing
```bash
python data_preprocessing.py
```
→ Generates `data/processed_data/cleaned_data.csv`

### 5. Run EDA
```bash
python eda.py
```
→ Generates 10 plots in `visuals/` folder

### 6. Train Models
```bash
python model_train.py
```
→ Trains XGBoost models, logs to MLflow, saves `.pkl` files in `models/`

### 7. View MLflow Dashboard (Optional)
```bash
mlflow ui
```
→ Open `http://127.0.0.1:5000` in browser to compare experiments

### 8. Launch Streamlit App
```bash
streamlit run app.py
```
→ Open `http://localhost:8501` in browser

---

## 📈 Model Performance

### Classification (Good Investment Prediction)
| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| Random Forest | 95.18% | 0.951 | 0.951 |
| **XGBoost** | **99.30%** | **0.993** | **0.993** |

### Regression (Future Price Prediction)
| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Random Forest | 55.74 | 40.21 | 0.927 |
| **XGBoost** | **13.67** | **10.98** | **0.995** |

✅ XGBoost selected as best model for both tasks.

---

## 🔍 EDA Highlights

1. Distribution of Property Prices
2. Size vs Price relationship & outliers
3. Price per SqFt by BHK & Furnished Status
4. Average Price by State & City (Top 10)
5. Correlation Heatmap
6. Nearby Schools & Hospitals vs Price
7. Availability Status distribution
8. Parking Space vs Price
9. Good Investment distribution & Public Transport vs Price
10. BHK vs Average Price

---

## 🧠 Target Variables

- **`Good_Investment`** → 1 if 2 out of 3 conditions met:
  - Price per SqFt ≤ median
  - Availability Status = Available
  - BHK ≥ 2

- **`Future_Price_5Y`** → `Price_in_Lakhs × (1.08)^5` (8% annual appreciation)

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
mlflow
streamlit
matplotlib
seaborn
joblib
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Deep Keni**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is for educational purposes as part of a Data Science capstone project.
```
