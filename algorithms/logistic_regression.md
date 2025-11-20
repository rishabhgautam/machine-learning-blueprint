## 📘 **Logistic Regression**

**Logistic Regression** is a classification algorithm that models the probability of class membership using the logistic (sigmoid) function. It is widely used for binary and multiclass classification due to its speed, simplicity, and interpretability.

## 1. Introduction
Predicts probabilities and maps them to class labels using a decision threshold.

## 2. Binary vs Multiclass Logistic Regression
- Binary Logistic Regression
- Softmax / Multinomial Logistic Regression

## 3. Sigmoid Function
```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

## 4. Hypothesis Function
```math
\hat{y} = \sigma(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)
```

## 5. Decision Boundary
```math
\hat{y} \geq 0.5 \Rightarrow class = 1
```

## 6. Cost Function (Log Loss)
```math
J = - \frac{1}{m} \sum [ y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) ]
```

## 7. Optimization (Gradient Descent)
```math
\beta_j := \beta_j - \alpha \cdot \frac{1}{m} \sum (\hat{y} - y)x_j
```

## 8. From-Scratch Implementation Structure
- Compute logits
- Apply sigmoid
- Compute log-loss
- Compute gradients
- Update weights

## 9. Regularization
- L1 (Lasso)
- L2 (Ridge)

## 10. Evaluation Metrics
- Accuracy
- Precision/Recall
- F1-score
- ROC-AUC
- PR-AUC

## 11. Assumptions of Logistic Regression
| Assumption | Meaning | Check Using |
|-----------|---------|-------------|
| Linearity in Log-Odds | Features relate linearly to log(p/(1-p)) | Box–Tidwell |
| Independence | Observations independent | Domain context |
| No Multicollinearity | Predictors not correlated | VIF |
| Large Sample Size | Stable MLE estimates | Data size |

## 12. When It Works Well
- Linearly separable classes
- Interpretable scenarios

## 13. When It Fails
- Non-linear boundaries
- Overlapping classes

## 14. Real-World Applications
- Fraud detection
- Churn prediction
- Default prediction

## 15. Scikit-Learn Example
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```
