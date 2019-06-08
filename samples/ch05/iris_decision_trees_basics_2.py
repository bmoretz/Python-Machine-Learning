import sys, csv
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from matplotlib.colors import  ListedColormap
from sklearn.tree import export_graphviz

iris_df = pd.read_csv("../data/iris.csv", low_memory=False)

features = ['sepal length', 'sepal width'] 

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

iris_df.iris.replace(targets, [0, 1, 2], inplace=True)

X = iris_df [features].values # matrix

y = iris_df ['iris'].values

plt.plot(X[y==0, 0], X[y==0, 1], 'r*', markersize=3, label="setosa")
plt.plot(X[y==1, 0], X[y==1, 1], 'go', markersize=2, label="versicolor")
plt.plot(X[y==2, 0], X[y==2, 1], 'bd', markersize=2, label="virginica")

criterion = "entropy"
max_depth=7
dt_clf = DecisionTreeClassifier(criterion = criterion, max_depth=max_depth)
dt_clf.fit(X, y)

export_graphviz(dt_clf, out_file='../images/dt_iris1.dot',feature_names=features, \
    class_names = ['setosa', 'versicolor', 'verginica'], filled=True, leaves_parallel=True)

x1, x2 = np.meshgrid(np.linspace(4, 8, 800).reshape(-1, 1), np.linspace(1, 5, 500).reshape(-1, 1),)
X_predict = np.c_[x1.ravel(), x2.ravel()]

#y_proba = dt_clf.predict_proba(X_predict)
y_predict = dt_clf.predict(X_predict)
print("X_predict", X_predict)
print("y_predict", y_predict)

z_fill = y_predict.reshape(x1.shape)
print(x1.shape)
custom_cmap = ListedColormap (['#33E3FF', '#FF5733', '#FFC300'])

plt.contourf(x1, x2, z_fill, cmap=custom_cmap, alpha = 0.4)


plt.xlabel(features[0])
plt.ylabel (features[1])
plt.legend (loc="best", frameon=False)
plt.grid(True)
plt.title('criterion = %s, max_depth= %i' %(criterion, max_depth))
    
plt.show()

export_graphviz(dt_clf, out_file='../images/dt_iris1.dot',feature_names=features, filled=True, leaves_parallel=True, impurity=True)
