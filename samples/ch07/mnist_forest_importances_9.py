import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

mnist = fetch_mldata('MNIST original', '../data')

def check_mnist_data (data, y, N):
    N_max = data.shape[0]
    for k in range(N):
        index = random.randint(1, N_max)
        image = data[index].reshape(28, 28)
        target = y [index]
    
        plt.subplot(5, 20, k + 1)
        plt.axis('off')
        plt.title('%i' % target, color='red')
        plt.imshow(image, cmap='Greys', interpolation = 'nearest')
    
    plt.show()
data, target  = mnist.data/255, mnist.target
X = data
y = target
'''
mask = (target == 9)  |  (target == 4) 
X = data[mask]
y = target[mask]
'''

#clf = ExtraTreesClassifier(n_estimators=50)
clf = RandomForestClassifier(n_estimators=50)
clf .fit(X, y)
#check_mnist_data (data, target, 100)
importances = clf .feature_importances_
print(importances)
importances = importances.reshape(28, 28)
plt.matshow(importances, cmap=plt.cm.hot)
plt.show()
