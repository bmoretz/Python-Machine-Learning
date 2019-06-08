import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler

def draw_marglin_lines (clf, xmin, xmax):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(xmin, xmax, 100)
    center_line = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = center_line - np.sqrt(1 + a ** 2) * margin
    yy_up = center_line + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, center_line, 'r-')
    plt.plot(xx, yy_down, 'r--')
    plt.plot(xx, yy_up, 'r--')

iris_df = pd.read_csv("../data/iris.csv", low_memory=False)
features = ['petal length', 'petal width'] 
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

iris_df.iris.replace(targets, [0, 1, 2], inplace=True)

X_original = iris_df [features].values # matrix
X = StandardScaler().fit_transform(X_original)
y = iris_df ['iris'].values

#reset target vlaue for binary classification
iris_df.iris.replace([0, 1, 2], targets, inplace=True)
iris_df.iris.replace(targets, [1, 0, 0], inplace=True)
Y = iris_df ['iris'].values

fignum = 1
xmin = ymin = -2
xmax = ymax = -xmin
for c_name, c in (('C = 1', 1.0), ('C = 100', 100.0)):
    clf = svm.LinearSVC(tol = 1e-4, C = c, dual=False)
    clf.fit(X, Y)

    draw_marglin_lines (clf, xmin, xmax)

    plt.plot(X[y==0, 0], X[y==0, 1], 'r*', markersize=3, label="setosa")
    plt.plot(X[y==1, 0], X[y==1, 1], 'go', markersize=2, label="versicolor")
    plt.plot(X[y==2, 0], X[y==1, 1], 'bd', markersize=2, label="verginica")
    plt.title("C (Penalty parameter) = " + str(c))

    XX, YY = np.meshgrid(np.linspace(xmin, xmax, 100).reshape(-1, 1), np.linspace(ymin, ymax, 100).reshape(-1, 1),)
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # draw sample divisions in color
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    cmap = plt.get_cmap()
    plt.pcolormesh(XX, YY, Z, cmap=cmap, alpha = 0.3)
    #plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    fignum = fignum + 1
    plt.legend(loc='best')
plt.show()
