# Machine Learning for Kyphosis Disease Classification

This project involves analyzing the Kyphosis dataset to understand the factors influencing the development of kyphosis after surgery and building machine learning models to predict its occurrence.

---

## 1. Understanding the Problem

Kyphosis is a spinal deformity characterized by an exaggerated forward curvature of the thoracic spine. It can lead to pain, stiffness, and neurological issues. Accurately predicting kyphosis post-surgery helps in preoperative planning and patient counseling. The goal is to identify key predictors and develop models that can classify patients at risk.

---

## 2. Data Import and Exploration

- **Libraries Used:** pandas, numpy, matplotlib, seaborn
- **Dataset:** `kyphosis.csv` with 81 records containing features `Age`, `Number`, `Start`, and target `Kyphosis`.

**Data Preview:**
```python
import pandas as pd
data = pd.read_csv("kyphosis.csv")
print(data.head())
```

**Sample Data:**
| Kyphosis | Age | Number | Start |
|-----------|------|---------|--------|
| absent    | 71   | 3       | 5      |
| absent    | 158  | 3       | 14     |
| present   | 128  | 4       | 5      |
| absent    | 2    | 5       | 1      |
| absent    | 1    | 4       | 15     |

---

## 3. Data Statistics and Visualization

### Descriptive Statistics
- Calculated mean, min, max ages using two methods:
  - **Method 1:** Using pandas `.mean()`, `.min()`, `.max()`
  - **Method 2:** Using Python's built-in functions `sum()` and `min()`, `max()`

### Data Visualizations
- **Correlation Heatmap:** Shows relationships among features.
- **Pairplot:** Visualizes feature interactions colored by `Kyphosis`.
- **Class Distribution:** Countplot indicating the percentage of patients with/without kyphosis (~21% with kyphosis).

---

## 4. Data Preprocessing

- **Label Encoding:** Convert `Kyphosis` labels (`absent`, `present`) into numeric (`0`, `1`) for modeling.
- **Feature and Target Separation:** Dropped `Kyphosis` from features, stored in `X`. Target stored in `y`.
- **Train-Test Split:** 80% training, 20% testing using `train_test_split`.
- **Feature Scaling:** Standardized features with `StandardScaler` for improved model performance.

---

## 5. Model Building and Evaluation

### a) Logistic Regression
- Trained on scaled data.
- **Performance:**
  - Training Accuracy: 84%
  - Testing Accuracy: 82%
- **Confusion Matrix:**

| Actual \ Predicted | 0 | 1 |
|---------------------|---|---|
| **0**             | 13 | 1 |
| **1**             | 3 | 0 |

- **Classification Report:**
  ```
                precision    recall  f1-score   support
    
                0       0.82      1.00      0.90       14
                1       0.00      0.00      0.00        3
                
        accuracy                           0.82       17
       macro avg       0.41      0.50      0.45       17
    weighted avg       0.68      0.82      0.74       17
  ```
  
*Note:* The model performs poorly in predicting positive cases (`Kyphosis=1`), with zero recall for class `1`.

---

### b) Decision Tree Classifier
- Hyperparameters tuned (`criterion='gini'`, `max_depth=5`, etc.).
- **Performance:**
  - Training Accuracy: ~76%
  - Testing Accuracy: ~76%
- **Confusion Matrix:**

| Actual \ Predicted | 0 | 1 |
|---------------------|---|---|
| **0**             | 13 | 1 |
| **1**             | 3 | 0 |

- **Classification Report:**
  ```
                precision    recall  f1-score   support
    
                0       0.81      0.93      0.87       14
                1       0.00      0.00      0.00        3
                
        accuracy                           0.76       17
       macro avg       0.41      0.46      0.43       17
    weighted avg       0.67      0.76      0.71       17
  ```

*Observation:* Similar to logistic regression, poor recall for class `1`.

---

### c) Random Forest Classifier
- Using 100 estimators.
- **Performance:**
  - Training Accuracy: 100% (overfitting)
  - Testing Accuracy: 82%
- **Confusion Matrix:**

| Actual \ Predicted | 0 | 1 |
|---------------------|---|---|
| **0**             | 13 | 1 |
| **1**             | 3 | 0 |

- **Classification Report:**
  ```
                precision    recall  f1-score   support
    
                0       0.82      1.00      0.90       14
                1       0.00      0.00      0.00        3
                
        accuracy                           0.82       17
       macro avg       0.41      0.50      0.45       17
    weighted avg       0.68      0.82      0.74       17
  ```

*Note:* Despite perfect training accuracy, it performs similarly on the test set with poor detection of positive cases.

---

## 6. Final Conclusions

| Model               | Train Accuracy | Test Accuracy | Test Confusion Matrix | Key Insights                                  |
|---------------------|------------------|-----------------|------------------------|----------------------------------------------|
| Logistic Regression | 84%             | 82%             | See above             | Fails to detect positive class; poor recall |
| Decision Tree       | ~76%            | ~76%            | See above             | Similar issues with class '1' detection     |
| Random Forest       | 100% (overfit)  | 82%             | Same as above         | Overfitting evident; class imbalance concern|

**Overall:**  
All models exhibit high overall accuracy but perform poorly in identifying `Kyphosis=1`. This indicates a need for techniques such as data balancing, feature engineering, or more sophisticated models to improve sensitivity.

---

## 7. Recommendations for Improvement
- Address class imbalance (e.g., SMOTE).
- Tune hyperparameters further.
- Explore additional features or domain-specific insights.
- Use cross-validation to optimize models.

---

## 8. Final Remarks
This analysis provides a foundation for understanding factors influencing kyphosis and highlights challenges in predictive modeling for imbalanced datasets. Future work can focus on improving recall for positive cases to enhance clinical utility.

---
