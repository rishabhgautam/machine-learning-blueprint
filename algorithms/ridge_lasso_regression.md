## 📘 **Ridge & Lasso Regression**

**Ridge Regression (L2)** and **Lasso Regression (L1)** are regularized linear regression techniques that improve generalization by penalizing large coefficients. They are especially useful in high-dimensional or multicollinear datasets.

## 1. Introduction
Regularization adds penalty terms to constrain model complexity.

## 2. Ridge Regression (L2)
```math
J = MSE + \lambda \sum \beta_j^2
```
- Shrinks coefficients
- Does not set them to zero
- Good for multicollinearity

## 3. Lasso Regression (L1)
```math
J = MSE + \lambda \sum |\beta_j|
```
- Can set coefficients to zero
- Performs feature selection

## 4. Elastic Net (L1 + L2)
```math
J = MSE + \lambda_1 \sum |\beta_j| + \lambda_2 \sum \beta_j^2
```

## 5. When to Use Which
| Method | Best For | Notes |
|--------|----------|-------|
| Ridge | Multicollinearity | Smooth shrinkage |
| Lasso | Sparse models | Feature selection |
| Elastic Net | Many correlated features | Balanced penalty |

## 6. Hyperparameter Tuning
- Choose lambda using CV
- Grid Search / Random Search

## 7. From-Scratch Implementation Structure
- Add penalty to loss
- Adjust gradients
- Update weights

## 8. Evaluation Metrics
- MSE, RMSE, MAE
- R² Score

## 9. Real-World Applications
- High-dimensional regression
- Text modeling
- Telecom churn feature selection

## 10. Scikit-Learn Examples
```python
from sklearn.linear_model import Ridge
Ridge(alpha=1.0).fit(X_train, y_train)
```

```python
from sklearn.linear_model import Lasso
Lasso(alpha=0.1).fit(X_train, y_train)
```

```python
from sklearn.linear_model import ElasticNet
ElasticNet(alpha=0.1, l1_ratio=0.5)
```
