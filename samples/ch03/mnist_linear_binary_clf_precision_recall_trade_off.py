import random, time

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np


data_home = '../data'
mnist = fetch_mldata('MNIST original', data_home)

def plot_scores (x, data, n_rows, n_columns, index, xlabel, ylabel, label, linestyle, marker): 
    #plt.ylim(0, 100)
    plt.subplot(n_rows, n_columns, index)
    plt.plot(x, data, label = label, linestyle = linestyle, marker = marker, markersize = 4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if (index == 1):
        plt.title('Total # of samples: %s' % (n))
    plt.legend(frameon=False)
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
thresholds = np.linspace(int(0.9 * np.min (y_scores)), int(0.9 * np.max (y_scores)), L)
print(thresholds)

accuracy, precision, recall, f1 = [], [], [], []

for l in range (L):    
    y_pred = (y_scores > thresholds[l])
    accuracy.append (np.mean(y_pred == y_test_9))
    precision.append (precision_score(y_test_9, y_pred))
    recall.append (recall_score(y_test_9, y_pred))
    f1.append (f1_score(y_test_9, y_pred))

n_rows = 2
n_columns = 2
index = 1

plot_scores (thresholds, precision, n_rows, n_columns, index, "Threshold", "Score(%)", "precision", '-', 'o')
plot_scores (thresholds, recall, n_rows, n_columns, index, "Threshold", "Score(%)", "recall", '--', 'D')
plot_scores (thresholds, accuracy, n_rows, n_columns, index + 1, "Threshold", "Score(%)", "accuracy", '-', 'o')
plot_scores (thresholds, f1, n_rows, n_columns, index + 1, "Threshold", "Score(%)", "f1", '--', 'D')
plot_scores (recall, precision, n_rows, n_columns, index + 2, "recall", "precision", "", '-', 'd')
    
print("total time: ", (time.time() - start0))
plt.show()