# Predictive-Modeling-of-Cardiovascular-Disease-Risk-

# Step 1 - Data Loading, Cleaning & Exploratory Data Analysis (EDA)

## Objectives
- Load and explore the Kaggle Heart Disease dataset.
- Perform basic data cleaning and preprocessing.
- Conduct initial EDA and visualize relationships.

## Steps Performed
1. Installed required libraries (pandas, seaborn, scikit-learn, etc.).
2. Loaded dataset into a DataFrame.
3. Removed duplicates and standardized column names.
4. Explored feature statistics and correlation.
5. Visualized:
   - Target distribution
   - Correlation heatmap
   - Feature pair relationships
6. Saved the cleaned dataset to `data/heart_cleaned.csv`.


# Step 2 - Data Preprocessing & Class Balancing (SMOTE)

## Objectives
- Prepare the dataset for model training.
- Encode categorical variables and scale numerical ones.
- Handle class imbalance using SMOTE.

## Steps Performed
1. Loaded cleaned data from Day 1.
2. Identified numerical and categorical features.
3. Applied preprocessing using `ColumnTransformer`:
   - Standardized numeric features.
   - One-hot encoded categorical variables.
4. Split data into training and testing sets.
5. Applied **SMOTE** to balance the target variable.
6. Saved:
   - Processed datasets (`X_train.npy`, `X_test.npy`, etc.)
   - Preprocessor pipeline (`preprocessor.pkl`).

# Step 3 - Model Training: Logistic Regression, Random Forest, LightGBM

## Objectives
- Train multiple machine learning models to predict cardiovascular disease risk.
- Evaluate baseline metrics and identify the best-performing model.

## Steps Performed
1. Loaded preprocessed data (`X_train`, `y_train`, `X_test`, `y_test`).
2. Trained three models:
   - Logistic Regression
   - Random Forest
   - LightGBM
3. Evaluated each model using:
   - Accuracy
   - F1 Score
   - AUC-ROC
4. Visualized model comparison and ROC curves.
5. Selected **LightGBM** as the best-performing model.
6. Saved trained models in the `model/` directory.

# Step 4 - Model Evaluation & Cross-Validation

## Objectives
- Evaluate model performance using multiple metrics.
- Perform k-fold cross-validation for robustness.
- Visualize ROC and Precision-Recall curves.

## Steps Performed
1. Loaded the best-performing model (LightGBM) and test data.
2. Computed key metrics:
   - Accuracy
   - F1 Score
   - Precision
   - Recall
   - AUC-ROC
3. Visualized:
   - Confusion Matrix
   - ROC Curve
   - Precision-Recall Curve
4. Performed **5-Fold Cross-Validation** for stability analysis.
5. Saved evaluation summary as `model/evaluation_summary.csv`.

# Step 5 - Explainable AI (XAI) using SHAP

## 🎯 Objective
On Day 5, the focus was on **interpreting the LightGBM model** to understand *why* it predicts cardiovascular disease risk.  
We implemented **Explainable AI (XAI)** techniques using **SHAP (SHapley Additive exPlanations)** to uncover which features contribute most to the model’s decisions.

---

## 🧠 Key Tasks Completed

### 1. Installed and Configured SHAP
- Installed the `shap` library in Colab for model interpretability.
- Initialized a `TreeExplainer` for the trained **LightGBM** model.

### 2. Handled Preprocessing Artifacts
- Addressed data shape mismatches caused by encoding and scaling.
- Reconstructed the feature names using:
  ```python
  feature_names = preprocessor.get_feature_names_out()
  ```
- Added fallback logic for cases when 'X_test' or 'X_train' were NumPy arrays:
```python
feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
```
- Ensured smooth integration between SHAP and LightGBM by converting arrays into properly named Pandas DataFrames.
  
### 3. Global Feature Importance
- Generated bar plots showing the mean absolute SHAP values for all features.
- These plots rank features by their overall influence on model output.
  
### 4. Summary Visualization
- Created summary plots showing each feature’s impact direction and distribution (positive = higher disease risk, negative = lower).
  
### 5. Individual Prediction Explanations
- Used force plots to interpret single predictions interactively:
```python
  shap.force_plot(explainer.expected_value, shap_values[i, :], X_sample.iloc[i, :])
```
- These show how individual feature values push the model toward “disease” or “no disease.”
  
### 6. Saved SHAP Insights
- Computed mean absolute SHAP values and saved them to:
```python
feature_importance_shap.csv
```
### 📊 Key Insights
- Features such as age, cholesterol, thalach (max heart rate), and oldpeak (ST depression) showed the strongest influence on cardiovascular disease predictions.
- SHAP visualizations provided both global and local interpretability — helping explain model behavior for both populations and individuals.

# Step 6 - Model Deployment with Flask

## 🎯 Objective
We deployed the trained **LightGBM cardiovascular disease model** using **Flask**, demonstrating how to serve real-time predictions through an API endpoint.

---

## 🚀 Key Tasks Completed

### ✅ 1. Model Serialization
- Saved the trained LightGBM model using `joblib` for later use in production.
- File generated: `lightgbm_heart_model.pkl`

### ✅ 2. Flask API Development
- Built a simple Flask app (`app.py`) exposing a `/predict` route.
- The API:
  - Accepts JSON input containing patient features.
  - Converts the input into a Pandas DataFrame.
  - Returns both the predicted class (`0` = No disease, `1` = Disease) and the risk probability.

### ✅ 3. Local Testing
- Used Python’s `requests` library to simulate POST requests.
- Verified correct JSON responses from the API.

Example request:
```python
sample_data = {
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 130,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```
### ✅ 4. (Optional) Dockerization
- Created a Dockerfile for easy deployment and containerization.
- Demonstrated building and running the container with:
```arduino
docker build -t heart-disease-api .
docker run -p 5000:5000 heart-disease-api
```
### 🏁 Project Complete
This project demonstrates end-to-end data science capability:
- Data ingestion, preprocessing, and modeling
- Class imbalance handling (SMOTE)
- Model explainability (SHAP)
- API deployment and serving with Flask
