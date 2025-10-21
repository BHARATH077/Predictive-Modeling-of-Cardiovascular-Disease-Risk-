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

## Folder Structure (After Day 2)
