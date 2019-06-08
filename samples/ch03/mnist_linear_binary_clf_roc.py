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
#mnist = fetch_mldata('MNIST original', '/Users/henryliu/mspc/ml_dev/ml_quantitative/data')

def plot_roc_curve (fpr, tpr, label): 
    plt.plot(fpr, tpr, label = label)
    plt.plot([0, 1], [0, 1], 'r--', label = "random")
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR (false positive rate)')
    plt.ylabel('TPR (true positive rate)')
    plt.legend(loc = "lower right", frameon=False)
    plt.grid(True) 
start0 = time.time()
data, target  = shuffle(mnist.data/255, mnist.target, random_state=0)

n = data.shape[0]
#n = 20000
X, y = data[0:n], target[0:n]

clf = SGDClassifier(average = True, tol = 1e-3)
rng = np.random.RandomState (None)
L = 24
split_ratio = 0.75
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, random_state=rng)
y_train_9 = (y_train == 9)
y_test_9 = (y_test == 9)
clf.fit(X_train, y_train_9) 
y_scores = clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test_9, y_scores)
plot_roc_curve(fpr, tpr, "SGD_A")
roc_auc_score = auc (fpr, tpr)
print('roc_auc_score: ', roc_auc_score)

#RandomForestClassifier
for n in range (1, 12, 2):
    rfc_clf = RandomForestClassifier (n_estimators = n)
    rfc_clf.fit(X_train, y_train_9)
    y_proba_rfc = rfc_clf.predict_proba (X_test)
    y_scores_rfc = y_proba_rfc
    
    fpr_rfc, tpr_rfc, _ = roc_curve(y_test_9, y_scores_rfc[:, 1])
    plot_roc_curve(fpr_rfc, tpr_rfc, "RandomForest n = " + str(n))
    roc_auc_score_rfc = auc (fpr_rfc, tpr_rfc)
    print('n roc_auc_score_rfc: ', n, roc_auc_score_rfc)
   
print("total time: ", (time.time() - start0))
plt.show()