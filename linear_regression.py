import numpy as np

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)


from sklearn.preprocessing import PolynomialFeatures
polynomial_features=PolynomialFeatures(degree=2, include_bias=False)
X_polynomial=polynomial_features.fit_transform(X)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_polynomial, y)
y_pred=lin_reg.predict(X_polynomial)

lin_reg.intercept_, lin_reg.coef_



from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val=train_test_split(X, y, test_size=.2)
    
    
    
    