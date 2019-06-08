import sys, csv

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd

iris_df = pd.read_csv("../data/iris.csv", low_memory=False)
'''
print("head\n", iris_df.head())
print("info\n", iris_df.info)
print("describe:\n", iris_df.describe())
'''
setosa = iris_df[iris_df.iris == "Iris-setosa"]
versicolor = iris_df[iris_df.iris == "Iris-versicolor"]
virginica = iris_df[iris_df.iris == "Iris-virginica"]
#print(setosa['sepal length'].values)
#print(setosa['sepal width'].values)
s = 9
plt.subplot(121)
plt.scatter (x = setosa['sepal length'].values, y = setosa['sepal width'].values, s=s, c = "r", marker="*", label="setosa")
plt.scatter (x = versicolor['sepal length'].values, y = versicolor['sepal width'].values, s=s, c = "g", marker="o", label="versicolor")
plt.scatter (x = virginica["sepal length"].values, y = virginica["sepal width"].values, s=s, c = "b", marker="d", label="verginica")
plt.xlabel("sepal length")
plt.ylabel ("sepal width")
plt.legend (loc="upper right")
plt.subplot(122)
plt.scatter (x = setosa['petal length'].values, y = setosa['petal width'].values, s=s, c = "r", marker="*", label="setosa")
plt.scatter (x = versicolor['petal length'].values, y = versicolor['petal width'].values, s=s, c = "g", marker="o", label="versicolor")
plt.scatter (x = virginica["petal length"].values, y = virginica["petal width"].values, s=s, c = "b", marker="d", label="verginica")
plt.xlabel("petal length")
plt.ylabel ("petal width")
plt.legend (loc="upper left")

attributes = ['sepal length', 'sepal width', 'petal length', 'petal width']
colors_palette = {'Iris-setosa': "red", 'Iris-versicolor': "green", 'Iris-virginica': "blue"}
groups = list(iris_df.iris)
colors = [colors_palette[c] for c in groups] 
scatter_matrix ( iris_df [attributes], figsize = (12, 8), alpha=0.8, color=colors, diagonal='kde')

plt.show()