import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


iris_df = pd.read_csv("../data/iris.csv", low_memory=False)

features = ['petal length', 'petal width'] 

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

iris_df.iris.replace(targets, [0, 1, 2], inplace=True)

X = iris_df [features].values # matrix
y = iris_df ['iris'].values

C = 10  # SVM regularization parameter
models = (svm.LinearSVC(C=C),
    svm.SVC(kernel='linear', C=C),
    svm.SVC(kernel='rbf', gamma=0.7, C=C),
    svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('LinearSVC', 'SVC(kernel=\'linear\')','SVC(kernel=\'rbf\')', 'SVC(kernel=\'poly 3\')')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.3, hspace=0.3)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for i, clf, title, ax in zip(np.arange(4), models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("BrBG"), alpha=0.5)
    ax.scatter(X0, X1, c=y, cmap=plt.get_cmap("autumn"), s=5)

    ax.text(0.5, 3, title, fontsize = 10)
    score = clf.score(X, y)
    print("score: ", i, score)
    if i == 0:
        ax.set_title("c=%s" %(C), fontsize=10, loc='left')
    if i == 2:
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
            
plt.show()
