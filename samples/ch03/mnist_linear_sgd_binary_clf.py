import random, time

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np

#mnist = fetch_mldata('MNIST original', data_home='/Users/henryliu/mspc/ml_dev/ml_quantitative/data')
mnist = fetch_mldata('MNIST original', '../data')


start0 = time.time()
data, target  = shuffle(mnist.data/255, mnist.target, random_state=0)
n = data.shape[0]
#n = 10000
X, y = data[0:n], target[0:n]
classifiers = [
    ("SGD", SGDClassifier(tol = 1e-3)),
    ("SGD_A", SGDClassifier(average = True, tol = 1e-3)),
    ("SVC", LinearSVC(tol = 1e-3)),
    ("LR", LogisticRegression(solver = 'sag', tol = 1e-1, C = 1.e4 / X.shape[0]))
]

test_dataset_ratio = [0.1, 0.25, 0.5, 0.75, 0.9]

rng = np.random.RandomState (None)
L = 6

for model, clf in classifiers:
    print("model %s" % model)
    start = time.time()
    total_time = []
    y_error_ratio_avg = []
    for split_ratio in test_dataset_ratio:
        y_error_ratio_l = []
        for l in range (L):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, random_state=rng)
            y_train_9 = (y_train == 9)
            y_test_9 = (y_test == 9)
            clf.fit(X_train, y_train_9) 
            y_pred = clf.predict(X_test)
            y_error_ratio_l.append (1 - np.mean(y_pred == y_test_9))
            print('l=%s' % l, y_error_ratio_l)
        
        y_error_ratio_avg.append(100 * np.mean(y_error_ratio_l))
        print(y_error_ratio_avg)
        total_time.append((time.time() - start))
    print("total time for ", model, ": ", (time.time() - start))

    plt.subplot(1, 2, 1)
    plt.plot(1. - np.array(test_dataset_ratio), y_error_ratio_avg, label = model)
    plt.scatter(1. - np.array(test_dataset_ratio), y_error_ratio_avg)
    plt.xlabel("Training set ratio")
    plt.ylabel("Test error (%)")
    plt.legend(loc = "upper right")
    plt.title('Total # of samples: %s' % (n))
    plt.legend(frameon=False)
    plt.subplot(1, 2, 2)
    plt.plot(1. - np.array(test_dataset_ratio), total_time, label = model)
    plt.scatter(1. - np.array(test_dataset_ratio), total_time)
    plt.xlabel("Training set ratio")
    plt.ylabel("Total time (seconds)")
    plt.legend(loc = "upper right")
    plt.legend(frameon=False)

print("total time: ", (time.time() - start0))
plt.show()
