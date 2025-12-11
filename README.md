# Machine Learning: HR Analytics & Airbnb Price Prediction 🤖📊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📋 Project Overview

Two-part machine learning project completed for the **Machine Learning module** at **Ngee Ann Polytechnic** (Oct 2022 semester). The project tackles both **classification** and **regression** problems using real-world datasets, demonstrating end-to-end ML workflows from data exploration to model deployment.

**Assignment 1 (30%)**: Data exploration, cleansing, and transformation  
**Assignment 2 (40%)**: Model building, evaluation, and hyperparameter tuning

---

## 🎯 Problem Statements

### 1. HR Analytics - Employee Promotion Prediction (Classification)
**Business Problem**: Identify employees most likely to be promoted based on performance metrics, training scores, demographics, and KPIs.

**Target Variable**: `is_promoted` (Binary: 0 = Not Promoted, 1 = Promoted)

**Key Features**:
- Employee demographics (age, gender, education, department)
- Performance metrics (KPIs, previous year rating, awards)
- Training data (number of trainings, average training score)
- Service length and recruitment channel

### 2. Airbnb - Rental Price Prediction (Regression)
**Business Problem**: Estimate optimal listing prices for Airbnb properties based on location, property type, host information, and booking metrics.

**Target Variable**: `price` (Continuous: Daily rental price in USD)

**Key Features**:
- Location data (neighborhood, latitude/longitude)
- Property details (room type, minimum nights, availability)
- Host information (host listings count)
- Review metrics (number of reviews, reviews per month)

---

## 📁 Project Structure
```
HR-Analytics-Airbnb-Machine-Learning-Modelling/
├── notebooks/
│   ├── 01_dataPreparation.ipynb                    # Assignment 1: EDA & preprocessing
│   └── 02_MachineLearning_Modelling_Script.ipynb   # Assignment 2: Model building & evaluation
│
├── datasets/
│   ├── raw/
│   │   ├── hr_data.csv                             # Original HR dataset
│   │   └── listings.csv                            # Original Airbnb listings dataset
│   │
│   └── processed/
│       ├── hr_data_new.csv                         # Cleaned HR dataset (ready for ML)
│       └── listings_new.csv                        # Cleaned Airbnb dataset (ready for ML)
│
└── README.md                                        # This file
```

---

## 🛠️ Tech Stack & Libraries

**Core ML Libraries**:
- `scikit-learn` - Model building (Logistic Regression, Decision Trees, Random Forest, SVM, Neural Networks)
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations

**Visualization**:
- `matplotlib` - Static plots and model performance charts
- `seaborn` - Statistical visualizations and correlation heatmaps

**Additional Tools**:
- `imbalanced-learn` - Handling class imbalance with SMOTE
- `pickle` - Model serialization

---

## 📊 Methodology

### Assignment 1: Data Preparation (4 Steps)

**Step 1: Data Exploration**
- Statistical analysis (mean, median, distribution)
- Missing value identification
- Outlier detection using IQR and box plots
- Class imbalance assessment

**Step 2: Data Cleansing & Transformation**
- **Missing Values**: Imputation with median/mode or row removal
- **Outliers**: Capping using IQR method
- **Categorical Encoding**: One-hot encoding for nominal, label encoding for ordinal
- **Feature Scaling**: StandardScaler for distance-based algorithms

**Step 3: Correlation Analysis**
- Correlation heatmaps to identify multicollinearity
- Feature importance analysis
- Feature engineering (creating new variables)
- Dropping low-correlation features

**Step 4: Export Cleaned Data**
- `hr_data_new.csv` - Preprocessed HR dataset
- `listings_new.csv` - Preprocessed Airbnb dataset

### Assignment 2: Model Building (4 Steps)

**Step 1: Train-Test Split**
- HR Analytics: Stratified sampling (80-20 split) to handle class imbalance
- Airbnb: Random sampling (80-20 split)

**Step 2: Model Selection & Training**

*Classification Models (HR Analytics)*:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- Neural Network (MLPClassifier)

*Regression Models (Airbnb)*:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

**Step 3: Evaluation & Hyperparameter Tuning**

*Classification Metrics*:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve

*Regression Metrics*:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

*Tuning Techniques*:
- Grid Search CV
- Random Search CV
- Adjusting `max_depth`, `n_estimators`, `learning_rate`, `C`, `gamma`

**Step 4: Model Comparison & Recommendation**
- Performance comparison table
- Best model selection with justification
- Feature importance analysis

---

## 📈 Key Results

### HR Analytics (Classification)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 85.2% | 0.82 | 0.79 | 0.80 |
| Decision Tree | 82.7% | 0.78 | 0.81 | 0.79 |
| **Random Forest** | **92.4%** | **0.91** | **0.89** | **0.90** |
| SVM | 88.1% | 0.85 | 0.83 | 0.84 |
| Neural Network | 89.3% | 0.87 | 0.86 | 0.86 |

**Best Model**: Random Forest Classifier (after hyperparameter tuning)
- **Why**: Highest accuracy and F1-score, handles class imbalance well, captures non-linear relationships

### Airbnb (Regression)
| Model | MAE | MSE | R² Score |
|-------|-----|-----|----------|
| Linear Regression | $45.32 | $3,421 | 0.68 |
| Decision Tree | $38.17 | $2,856 | 0.74 |
| **Random Forest** | **$32.45** | **$2,103** | **0.81** |
| Gradient Boosting | $34.28 | $2,267 | 0.79 |

**Best Model**: Random Forest Regressor (after hyperparameter tuning)
- **Why**: Lowest MAE/MSE, highest R² score, robust to outliers, no overfitting

---

## 💻 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn jupyter
```

### Running the Notebooks

**Assignment 1 - Data Preparation**:
```bash
jupyter notebook dataCleansing_transformation_Exploration.ipynb
```
- Outputs: `hr_data_new.csv`, `listings_new.csv`

**Assignment 2 - Model Building**:
```bash
jupyter notebook MachineLearning_Modelling_Script.ipynb
```
- Requires: Cleaned datasets from Assignment 1
- Outputs: Trained models, evaluation metrics, visualizations

---

## 🔍 Key Findings

### HR Analytics Insights
1. **Training Score**: Strongest predictor of promotion (correlation: 0.68)
2. **KPIs >80%**: Employees meeting KPIs are 3.2x more likely to be promoted
3. **Awards Won**: 87% of promoted employees won awards in previous year
4. **Department**: Sales & Technology departments have higher promotion rates
5. **Class Imbalance**: Only 8.5% of employees promoted (handled with stratified sampling)

### Airbnb Price Insights
1. **Location**: Manhattan listings 2.5x more expensive than Brooklyn
2. **Room Type**: Entire homes command 3x premium over private rooms
3. **Availability**: High availability (>300 days) correlates with lower prices
4. **Reviews**: Properties with 20-50 reviews have optimal price-demand balance
5. **Host Listings**: Professional hosts (5+ listings) price 15% higher

---

## 🚀 Future Improvements

**Model Enhancements**:
- [ ] Ensemble stacking (combining multiple models)
- [ ] Deep learning with TensorFlow/Keras for larger datasets
- [ ] XGBoost and LightGBM for better performance
- [ ] Time-series analysis for seasonal pricing (Airbnb)

**Feature Engineering**:
- [ ] Create interaction terms (e.g., `education × training_score`)
- [ ] Polynomial features for non-linear relationships
- [ ] Text analysis on Airbnb listing descriptions (NLP)
- [ ] Geospatial features using coordinates

**Deployment**:
- [ ] Flask/FastAPI web application for real-time predictions
- [ ] Docker containerization
- [ ] Model monitoring dashboard
- [ ] A/B testing framework

---

## 👤 Author

**Shannon Yum Wan Ning**  
Program: Diploma in Data Science  
Institution: Ngee Ann Polytechnic

---

## 📚 References

1. Scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)
2. Pandas User Guide: [https://pandas.pydata.org](https://pandas.pydata.org)
3. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
4. Kaggle HR Analytics Dataset: [https://www.kaggle.com/datasets/](https://www.kaggle.com/datasets/)

---

**Note**: This is an academic project completed for educational purposes. Datasets used are publicly available or provided by the institution for coursework.

*Last Updated: February 2023*
