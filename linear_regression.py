

import numpy as np
import matplotlib.pyplot as plt

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
    training_error, val_error = [], []
    for k in range(1, len(X_train)):
        print(k)
        model.fit(X_train[:k], y_train[:k])
        y_pred_train=model.predict(X_train[:k])
        y_pred_val=model.predict(X_val)
        training_error.append(mean_squared_error(y_pred_train, y_train[:k]))
        val_error.append(mean_squared_error(y_pred_val, y_val))
    plt.plot(np.sqrt(training_error), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_error), "b-+", linewidth=2, label="val")


def plot_learning_curves1(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        print(m)
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    print(val_errors)
    
lin_reg = LinearRegression()
plot_learning_curves1(lin_reg, X, y)

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)


from sklearn.pipeline import Pipeline

poly_features=Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("linear", LinearRegression()), 
        ])
        
plot_learning_curves1(poly_features, X, y)    


from sklearn.pipeline import Pipeline

polynomial_regression4 = Pipeline([
        ("poly_features", PolynomialFeatures(degree=5, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

plot_learning_curves1(polynomial_regression4, X, y)


from sklearn.linear_model import Ridge
ridge_reg=Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X,y)
ridge_reg.predict([[1.55]])

