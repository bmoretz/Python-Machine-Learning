import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

data_home = '../data/'
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
n_samples = 2000
rfc_clf = RandomForestClassifier (n_estimators = n_estimators)
rfc_clf.fit(X_train, y_train)
y_pred_rfc = rfc_clf.predict (X_test[0:n_samples])
print(y_pred_rfc)

#check_mnist_multiclass_data (X_test[0:n_samples], y_test[0:n_samples], y_pred_rfc, 1, 20, "./multiclass/" + "multiclass_conf_matrix.jpg")
print(np.sort(y_test[0:n_samples]))
conf_matrix = confusion_matrix (y_test[0:n_samples], y_pred_rfc)
'''
print(conf_matrix)
plt.matshow (conf_matrix)
'''
row_sums = conf_matrix.sum(axis = 1, keepdims = True)
normalized_conf_matrix = conf_matrix / row_sums
np.fill_diagonal (normalized_conf_matrix, 0)
np.set_printoptions(precision=3)
print(normalized_conf_matrix)
plt.matshow (normalized_conf_matrix)
x_data = np.sort (np.unique(y_pred_rfc))
print("total time: ", (time.time() - start0))
plt.show()