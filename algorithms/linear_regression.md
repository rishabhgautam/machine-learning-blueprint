## 📘 **Linear Regression**

**Linear Regression** is a supervised learning algorithm used to model the linear relationship between one or more input features and a continuous target variable. It is simple, interpretable, and serves as the foundation for most statistical and ML regression methods.

## 1. Introduction
Linear Regression fits a straight line (or hyperplane) that best captures the relationship between predictors and the outcome variable through error minimization.

## 2. Simple vs Multiple Linear Regression
- Simple → One predictor
- Multiple → Multiple predictors

## 3. Mathematical Formulation
```math
\hat{y} = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n
```

## 4. Cost Function (RSS / MSE)
```math
RSS = \sum (y - \hat{y})^2
```
```math
MSE = \frac{1}{m} \sum (y - \hat{y})^2
```

## 5. Optimization Approaches
- Normal Equation
- Gradient Descent (Batch, SGD, Mini-Batch)

## 6. From-Scratch Implementation Structure
- Initialize parameters
- Predict
- Compute cost
- Compute gradients
- Update parameters

## 7. Feature Engineering
- Scaling
- Polynomial features
- One-hot encoding
- Outlier treatment

## 8. Regularization Extensions
- Ridge
- Lasso
- Elastic Net

## 9. Evaluation Metrics
- MSE, RMSE, MAE
- R², Adjusted R²

## 10. Residual Diagnostics
Check for:
- Non-linearity
- Heteroscedasticity
- Outliers

## 11. Assumptions of Linear Regression
| Assumption | Meaning | Check Using |
|-----------|---------|-------------|
| Linearity | Relationship is linear | Residual plot |
| Independence | Errors independent | Durbin-Watson |
| Homoscedasticity | Constant variance | Breusch–Pagan |
| Normality | Errors normal | Q-Q plot |
| No Multicollinearity | Predictors uncorrelated | VIF |

## 12. Multicollinearity & VIF
```math
VIF = \frac{1}{1 - R_j^2}
```

## 13. When It Works Well
- Approximate linear trend
- Low multicollinearity

## 14. When It Fails
- Non-linear patterns
- Outliers
- Correlated predictors

## 15. Real-World Applications
- Forecasting
- Pricing
- Attribution modeling

## 16. Scikit-Learn Example
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```
