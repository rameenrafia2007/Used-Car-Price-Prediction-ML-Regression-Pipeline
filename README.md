# Used-Car-Price-Prediction-ML-Regression-Pipeline

### 📋 Overview

An end-to-end machine learning regression pipeline built to predict used car prices. The project covers the complete ML workflow — from data cleaning and preprocessing to model training, hyperparameter tuning, evaluation, and deployment-ready serialization.

### 🎯 Objectives

Perform exploratory data analysis on the used_cars.csv dataset
Handle missing values, encode categorical variables, and scale numeric features
Train and compare multiple regression architectures: Linear Regression, Polynomial Features, Ridge, and Lasso
Use cross-validated R² scores to select the best-performing model while keeping the test set untouched
Serialize the final model for future deployment


### 🛠️ Tools & Libraries


Python, Pandas, NumPy
Matplotlib, Seaborn — visualization
Scikit-learn — train_test_split, StandardScaler, LinearRegression, RidgeCV, LassoCV, PolynomialFeatures, KFold, cross_val_score
Joblib — model serialization


### 🔍 Methodology


EDA — Explored dataset structure, visualized the target variable (price) distribution, and generated a correlation heatmap of numerical features
Preprocessing

Dropped non-predictive metadata columns (id, link) and a column with 98% missing values (condition)
Imputed missing values in speed_levels using the median
One-hot encoded categorical variables
Applied an 80/20 train-test split and StandardScaler to normalize features and prevent data leakage



Model Training & Tuning — Trained a baseline Linear Regression, Degree-2 Polynomial Regression, and cross-validated Ridge and Lasso models to search for optimal regularization strength
Model Comparison — Compared all models using mean cross-validated R² scores. Found that Ridge Regression without polynomial expansion was the most stable architecture — the polynomial models overfit badly due to the small dataset size relative to feature count (39 columns)
Final Evaluation — Evaluated the winning model on the untouched test set and exported it via joblib


###  📊 Results

MetricScoreSelected ModelRidge Regression (No Polynomial Expansion)Test Set R² Score0.2890 (28.90%)Test Set RMSE$2,074.15

Key takeaway: With a small dataset and high-dimensional one-hot encoded features, simpler regularized models (Ridge) generalized far better than complex polynomial expansions, which overfit the training data.

### 📁 Repository Structure

├── WeekAssignment2_RAFIA_RAMEEN.ipynb   # Main analysis & modeling notebook
├── used_cars_model.pkl                   # Serialized final Ridge Regression model
└── README.md

### 🚀 How to Run


Clone this repository
Open WeekAssignment2_RAFIA_RAMEEN.ipynb in Jupyter Notebook or Google Colab
Ensure used_cars.csv is in the same directory
Run all cells sequentially to reproduce the pipeline
To use the saved model directly:


python   import joblib
   model = joblib.load('used_cars_model.pkl')
   predictions = model.predict(X_new_scaled)


Part of my AI/ML internship assignments — focused on building a complete, evaluated, and deployable regression pipeline.
