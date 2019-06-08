import sys, csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

def plot_logit (x, y_proba, n_rows, n_columns, index, feature, target): 
    #plt.ylim(0, 100)
    plt.subplot(n_rows, n_columns, index)
    plt.plot(x, y_proba[:, 1], "g-", label=target)
    plt.plot(x, y_proba[:, 0], "r--", label='Not %s' %(target))
    
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xlabel(feature)
    plt.ylabel ("probability")
    plt.legend (loc="best")
    plt.grid(True)

iris_df = pd.read_csv("../data/iris.csv", low_memory=False)

features = ['sepal length', 'petal length'] 
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
n_rows = 2
n_columns = 3
index = 1
for feature in features:
    for target in targets:
        print("feature, target:", feature, target)
        X = iris_df [[feature]].values # matrix
        y = (iris_df ['iris'] == target).values.astype(np.int) #1d array
    
        log_reg_clf = LogisticRegression()
        log_reg_clf.set_params(verbose=True)
        log_reg_clf.fit (X, y)
        print(log_reg_clf.get_params(deep=True))
        X_predict = np.linspace (0, 10, 1000).reshape (-1, 1)
        y_proba = log_reg_clf.predict_proba(X_predict)
        plot_logit(X_predict, y_proba, n_rows, n_columns, index, feature, target)
        index = index + 1
plt.show()