import sys, csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from matplotlib.colors import  ListedColormap

iris_df = pd.read_csv("../data/iris.csv", low_memory=False)

features = ['petal length', 'petal width'] 
#features = ['sepal length', 'sepal width'] 
#features = ['petal length', 'sepal width'] 
#features = ['sepal length', 'petal width'] 

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

iris_df.iris.replace(targets, [0, 1, 2], inplace=True)

X = iris_df [features].values # matrix
y = iris_df ['iris'].values
#print(X, y)
plt.plot(X[y==0, 0], X[y==0, 1], 'r*', markersize=3, label="setosa")
plt.plot(X[y==1, 0], X[y==1, 1], 'go', markersize=2, label="Iversicolor")
plt.plot(X[y==2, 0], X[y==2, 1], 'bd', markersize=2, label="virginica")

softmax_reg_clf = LogisticRegression()
softmax_reg_clf.set_params(multi_class="multinomial", solver="lbfgs")
#softmax_reg_clf.set_params(multi_class="multinomial", solver="newton-cg", C=0.01)
#softmax_reg_clf.set_params(multi_class="multinomial", solver="sag")

softmax_reg_clf.fit(X, y)

x1, x2 = np.meshgrid(np.linspace(0, 8, 800).reshape(-1, 1), np.linspace(0, 5, 500).reshape(-1, 1),)
X_predict = np.c_[x1.ravel(), x2.ravel()]

y_proba = softmax_reg_clf.predict_proba(X_predict)
y_predict = softmax_reg_clf.predict(X_predict)
print("X_predict", X_predict)
print("y_proba", y_proba)
print("y_predict", y_predict)

z_fill = y_predict.reshape(x1.shape)
z_densities = y_proba[:, 1].reshape(x1.shape)

custom_cmap = ListedColormap (['#33E3FF', '#FF5733', '#FFC300'])

plt.contourf(x1, x2, z_fill, cmap=custom_cmap)
contour = plt.contour(x1, x2, z_densities, cmap=plt.cm.brg)
plt.clabel(contour, inline = 1, fontsize = 10)

plt.xlabel(features[0])
plt.ylabel (features[1])
plt.legend (loc="best", frameon=False)
plt.grid(True)
    
plt.show()
