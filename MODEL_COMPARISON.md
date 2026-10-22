# Baseline Classification Pipeline — Telco Customer Churn

## 1. Project Overview

The objective of this project is to build and evaluate a baseline classification pipeline that predicts whether a telecommunications customer will leave the company (churn). The project compares a majority-class baseline with two Logistic Regression models to understand how class imbalance affects classification performance.

The models evaluated are:

1. Majority-Class Baseline
2. Logistic Regression (Default)
3. Logistic Regression (Balanced)

## 2. Dataset Overview

The dataset used is `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

- **Total records:** 7,043
- **Total original columns:** 21
- **Target variable:** `Churn`
- **No churn:** 5,174 customers (73.46%)
- **Churn:** 1,869 customers (26.54%)

The `customerID` column was excluded because it is an identifier rather than a useful predictive feature. The `TotalCharges` column was converted to a numeric data type, with missing values handled during preprocessing.

The dataset is imbalanced because customers who do not churn represent approximately 73.46% of the records.

## 3. Train-Test Split

The dataset was divided into training and testing sets using an 80/20 split, with stratification to preserve the original churn distribution.

- **Training records:** 5,634
- **Testing records:** 1,409
- **Training features:** 19
- **Testing features:** 19

The target variable was separated from the input features before splitting. Stratification was applied using `stratify=y`.

Splitting the data before fitting preprocessing steps helps prevent data leakage from the test set into model training.

## 4. Data Preprocessing

A preprocessing pipeline was created using `ColumnTransformer` and Scikit-learn pipelines.

### Numerical Features

The four numerical features were:

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Numerical preprocessing included:

- Missing-value imputation using the median.
- Feature scaling using `StandardScaler`.

### Categorical Features

The 15 categorical features included `gender`, `Partner`, `Dependents`, `Contract`, `PaymentMethod`, and other customer service and subscription attributes.

Categorical preprocessing included:

- Missing-value imputation using the most frequent value.
- One-hot encoding using `OneHotEncoder(handle_unknown="ignore")`.

The preprocessing steps were included within each model pipeline so that transformations were fitted on the training data and then applied to the test data.

## 5. Models Evaluated

### 5.1 Majority-Class Baseline

The baseline used `DummyClassifier(strategy="most_frequent")`. It always predicted class `0`, representing customers who do not churn.

This model provides a simple reference point for determining whether the more advanced models provide meaningful improvements.

### 5.2 Logistic Regression (Default)

The first Logistic Regression model used the default class weighting. The maximum number of iterations was set to 2,000, and the random state was set to 42.

This model was evaluated using the same training and testing data as the other models.

### 5.3 Logistic Regression (Balanced)

The second Logistic Regression model used `class_weight="balanced"`.

This setting adjusts the model's class weights to account for the unequal number of churn and non-churn examples. It is intended to help the model identify the minority class more effectively.

## 6. Evaluation Metrics

The models were evaluated using the following metrics:

- **Accuracy:** The proportion of all predictions that were correct.
- **Precision:** The proportion of predicted churners who actually churned.
- **Recall:** The proportion of actual churners correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model distinguishes between the two classes across different classification thresholds.

Recall is particularly important in churn prediction because failing to identify a customer who is likely to leave may result in a lost business opportunity.

## 7. Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority-Class Baseline | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression (Default) | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| Logistic Regression (Balanced) | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 |

### Results Interpretation

The majority-class baseline achieved 73.46% accuracy. However, its precision, recall, and F1-score for the churn class were all zero because it predicted every customer as a non-churner. Its ROC-AUC was 0.5000, indicating no discriminatory ability when using its constant probability scores.

The default Logistic Regression model achieved the highest accuracy at 80.55%. Its precision was 65.72%, meaning that approximately 65.72% of the customers it predicted would churn actually churned. Its recall was 55.88%, so it identified slightly more than half of the actual churners.

The balanced Logistic Regression model achieved 78.34% recall, substantially higher than the default model's 55.88%. It identified more actual churners but had lower precision at 50.43% and lower accuracy at 73.81%.

The balanced model also achieved a slightly higher F1-score of 0.6136 compared with 0.6040 for the default model. The default model had a marginally higher ROC-AUC of 0.8419 compared with 0.8413 for the balanced model.

Overall, the results demonstrate that changing class weights can improve churn detection at the cost of additional false-positive predictions.

## 8. Confusion Matrix Analysis

The confusion matrices use rows for actual classes and columns for predicted classes. Class `0` represents no churn, while class `1` represents churn.

### Majority-Class Baseline

```text
[[1035    0]
 [ 374    0]]
```

- True negatives: 1,035
- False positives: 0
- False negatives: 374
- True positives: 0

The baseline correctly identified all non-churners but failed to identify any churners.

### Logistic Regression (Default)

```text
[[926 109]
 [165 209]]
```

- True negatives: 926
- False positives: 109
- False negatives: 165
- True positives: 209

The default model correctly identified 209 churners but missed 165 actual churners. It also incorrectly flagged 109 non-churning customers as churners.

### Logistic Regression (Balanced)

```text
[[747 288]
 [ 81 293]]
```

- True negatives: 747
- False positives: 288
- False negatives: 81
- True positives: 293

The balanced model correctly identified 293 churners and missed only 81 actual churners. However, it incorrectly flagged 288 non-churning customers as churners.

### Confusion Matrix Comparison

The balanced model identified 84 more actual churners than the default model and reduced false negatives from 165 to 81. However, it also increased false positives from 109 to 288.

This demonstrates the practical trade-off between finding more potential churners and avoiding unnecessary customer-retention interventions.

## 9. Business Interpretation

A telecommunications company can use churn prediction to identify customers who may cancel their subscriptions and offer them suitable retention measures.

The default Logistic Regression model is preferable when the company wants a more precise list of customers to contact and has limited resources for retention campaigns. Its higher precision means that a larger proportion of flagged customers are actual churners.

The balanced Logistic Regression model is preferable when missing a potential churner is more costly than contacting a customer who would have stayed anyway. It detects more churners and has fewer false negatives, making it useful when the company prioritizes customer retention.

However, false positives also have costs. Unnecessary discounts, calls, and promotional offers can consume resources and reduce profitability. The company should therefore choose the model based on its retention budget and the relative costs of missed churners and false alarms.

The majority-class baseline is unsuitable for an operational churn-detection system because it never identifies customers who churn.

## 10. Conclusion

This project demonstrates how to build a classification pipeline that combines preprocessing and Logistic Regression while avoiding data leakage from the test set.

The default Logistic Regression model achieved the highest accuracy and precision. The balanced Logistic Regression model achieved the highest recall and F1-score among the three evaluated models.

The results show that accuracy alone is insufficient for evaluating a model on an imbalanced dataset. Precision, recall, F1-score, ROC-AUC, and confusion matrices provide a more complete understanding of model performance.

For a business that prioritizes identifying as many potential churners as possible, the balanced model is a strong candidate. The final choice should depend on the financial impact of false negatives and false positives.

## 11. Limitations and Future Improvements

- Evaluate different classification thresholds to adjust the precision-recall trade-off.
- Compare additional classification algorithms, such as Random Forest or gradient boosting.
- Use cross-validation to assess how consistently the models perform.
- Consider the financial costs of retention campaigns and customer churn when selecting a final model.
- Evaluate whether model performance remains stable on newer customer data.

The current results provide a baseline for further experimentation rather than a final production-ready churn prediction system.