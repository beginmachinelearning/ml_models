import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
plt.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')


iris = datasets.load_iris()
X_whole= iris["data"]
y_whole= iris["target"]
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

y_iris_setosa=np.where(y_whole==0)
X_iris_setosa=X_whole[y_iris_setosa,:].reshape(50,4)



y_iris_versicolor=np.where(y_whole==1)
X_iris_versicolor=X_whole[y_iris_versicolor,:].reshape(50,4)

y_iris_virginica=np.where(y_whole==2)
X_iris_virginica=X_whole[y_iris_virginica,:].reshape(50,4)

ifg=plt.figure(figsize=(12,5))
first_plot=ifg.add_subplot(121)
first_plot.scatter(X_iris_setosa[:, 2], X_iris_setosa[:,3], c='g')
first_plot.scatter(X_iris_versicolor[:,2], X_iris_versicolor[:,3], c='r')
first_plot.scatter(X_iris_virginica[:,2], X_iris_virginica[:,3], c='m')

houseprice_year=np.array([[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010], 
                          [169191,182863,198087,224567,249191,276221,305637,322634,305269,242033,228268]], np.int32)
                          
houseprice_year=np.transpose(houseprice_year)  


figure1=plt.figure(figsize=(10, 5))
first_plot=figure1.add_subplot(121)
first_plot.scatter(X_iris_setosa[:, 2], X_iris_setosa[:,3], c='g')


first_plot.scatter(X_iris_virginica[:, 2], X_iris_virginica[:,3], c='r')
plt.legend(['setosa', 'versicolor', 'virginica'])
plot_iris.show()


svc_clf=Pipeline(
        [
         ("scaler", StandardScaler()),
         ("linear_svc", LinearSVC(C=1, loss="hinge"))
                ]
        )

svc_clf.fit(X, y)

y_pred=svc_clf.predict([[2.1, 3.55]])


from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ])

polynomial_svm_clf.fit(X, y)

import pandas as pd


dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=";")