import time

from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np

data_home = '../data'
mnist = fetch_mldata('MNIST original', data_home)

def check_mnist_multiclass_data (data, y, y_pred, m_row, n_column, filename):
    N_max = data.shape[0]
    N = m_row * n_column
    for index in range(N_max):
        image = data[index].reshape(28, 28)
        target = y [index]
        pred = y_pred [index]
        plt.subplot(m_row, n_column, index + 1)
        plt.axis('off')
        plt.title('%i\n%i' %(target, pred), color='red')
        plt.imshow(image, cmap='Greys', interpolation = 'nearest')
 
    plt.savefig (filename)
    
start0 = time.time()
data, target  = shuffle(mnist.data/255, mnist.target, random_state=0)

n = data.shape[0]

X, y = data[0:n], target[0:n]

rng = np.random.RandomState (None)

split_ratio = 0.75
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, random_state=rng)
#y_train_9 = (y_train == 9)
#y_test_9 = (y_test == 9)

#RandomForestClassifier
n_estimators = 12
n_samples = 20
rfc_clf = RandomForestClassifier (n_estimators = n_estimators)
rfc_clf.fit(X_train, y_train)
y_pred_rfc = rfc_clf.predict (X_test[0:n_samples])
print(y_pred_rfc.astype(int))

check_mnist_multiclass_data (X_test[0:n_samples], y_test[0:n_samples], y_pred_rfc, 1, 20, "./multiclass/" + "multiclass.jpg")

print("total time: ", (time.time() - start0))
#plt.show()