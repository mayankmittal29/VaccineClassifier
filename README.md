# 🦠 Flu Vaccine Prediction Project 🌡️

## Problem Description

The goal of this project is to predict the likelihood of individuals receiving two types of flu vaccines: xyz and seasonal. The predictions should output probabilities (float values between 0.0 and 1.0) for each respondent's probability of receiving each vaccine.

## Dataset Description 📊

- **Features:** There are 36 columns including respondent_id and 35 features:
  - Various behavioral, opinion-based, demographic, and socio-economic factors.
  - Features include xyz_concern, xyz_knowledge, behavioral patterns, doctor recommendations, health conditions, opinions on vaccine effectiveness and risks, demographics (age, education, race, sex), income, marital status, housing situation, employment details, and geographic region.

- **Labels:**
  - xyz_vaccine: Binary (0 = No, 1 = Yes) indicating if the respondent received the xyz flu vaccine.
  - seasonal_vaccine: Binary (0 = No, 1 = Yes) indicating if the respondent received the seasonal flu vaccine.

## Evaluation Metric 📏

The model performance will be evaluated using the area under the Receiver Operating Characteristic curve (ROC AUC) for each target variable (xyz_vaccine and seasonal_vaccine). The final score will be the mean ROC AUC across both targets.

## Approach 📝

- **Data Exploration and Preprocessing:**
  - Explore data distributions, handle missing values, and preprocess categorical variables (encoding, scaling, etc.).

- **Feature Engineering:**
  - Extract useful features from all variables/features.

- **Model Selection:**
  - Experiment with various machine learning models suitable for binary classification (e.g., Logistic Regression, Random Forest, SVM, Naive Bayes, K-Nearest Neighbors).
  - Utilize techniques like cross-validation for model selection and hyperparameter tuning.

- **Training and Validation:**
  - Train models on the training dataset, validate on a separate validation set (or using cross-validation).
  - Optimize models to maximize ROC AUC score.

- **Prediction and Submission:**
  - Generate predictions on the test set.
  - Prepare submission files with respondent_id, predicted probabilities for xyz_vaccine, and seasonal_vaccine.

## Tools and Libraries 🛠️

- **Python Libraries:** Pandas, NumPy, scikit-learn for data manipulation, modeling, and evaluation.
- **Visualization:** Matplotlib, Seaborn for data exploration and result visualization.
- **Model Evaluation:** `sklearn.metrics.roc_auc_score` for evaluating ROC AUC.

## Files Included 📄

- `training_set_features.csv`: Training dataset containing respondent information.
- `training_set_labels.csv`: Contains vaccine uptake labels.
- `test_set_features.csv`: Test dataset for making predictions.
- `predictions_vacc.csv`: Template for submitting predictions in the required format.

## Results and Conclusion 📈

- Evaluate model performance based on ROC AUC scores.
- Consider additional techniques and exploratory data analysis (EDA) to further improve model performance.

