import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import KernelPCA

iris_df = pd.read_csv("../data/iris.csv", low_memory=False)
features = ['sepal length', 'sepal width', 'petal length', 'petal width'] 

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
target_names = targets
iris_df.iris.replace(targets, [0, 1, 2], inplace=True)

X = iris_df [features].values # matrix
y = iris_df ['iris'].values

kpca = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=False, gamma=.5)
X_r = kpca.fit(X).transform(X)

# Percentage of variance explained for each components
#print('explained variance ratio with the first two components: %s for features %s'
#    % (str(kpca.explained_variance_ratio_), features))

colors = ['r', 'g', 'b']
markers = ["*", 'o', 'd']
lw = 2

for color, marker, i, target_name in zip(colors, markers, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, marker=marker, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('2-component kernel PCA of IRIS dataset with all 4 features')

plt.show()
