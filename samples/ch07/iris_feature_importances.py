from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

clf = ExtraTreesClassifier(n_estimators=100)
clf = clf.fit(X, y)
print(iris["feature_names"], "\n", clf.feature_importances_ )