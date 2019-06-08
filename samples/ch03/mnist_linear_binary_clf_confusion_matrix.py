import random, time

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np


data_home = '../data'
#mnist = fetch_mldata('MNIST original', data_home=my_data_home)
mnist = fetch_mldata('MNIST original', data_home)

def check_mnist_data (data, y, m_row, n_column, filename):
    N_max = data.shape[0]
    N = m_row * n_column
    for k in range(N):
        index = random.randint(1, N_max) - 1
        image = data[index].reshape(28, 28)
        target = y [index]
    
        plt.subplot(m_row, n_column, k + 1)
        plt.axis('off')
        plt.title('%i' % target, color='red')
        plt.imshow(image, cmap='Greys', interpolation = 'nearest')

    plt.savefig (filename)
    
start0 = time.time()
data, target  = shuffle(mnist.data/255, mnist.target, random_state=0)

n = data.shape[0]
#n = 20000
X, y = data[0:n], target[0:n]

classifiers = [
    ("SGD", SGDClassifier(tol = 1e-3)),
    ("SGD_A", SGDClassifier(average = True, tol = 1e-3)),
    ("LR", LogisticRegression(solver = 'sag', tol = 1e-1, C = 1.e4 / X.shape[0]))
]

test_dataset_ratio = [0.25, 0.5, 0.75]

rng = np.random.RandomState (None)
L = 3

for model, clf in classifiers:
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

            if l == (L - 1):
                print("model, split_ratio, l ", model, split_ratio, l)
                filename_pre_fix = './figures/' + model + '_' + str(split_ratio) + '_' + str(l) 
                check_mnist_data (X_test, y_test, 4, 20, filename_pre_fix + '_original_testing_data.jpg')
                check_mnist_data (X_test, y_test_9, 4, 20, filename_pre_fix + '_binarized_testing_data.jpg')
                check_mnist_data (X_test, y_pred, 4, 20, filename_pre_fix + '_predictions.jpg') 
                print('at l=%s error ratio:' % l, y_error_ratio_l)
                conf_matrix = confusion_matrix(y_test_9, y_pred)            
                print('at l=%s confusion matrix:\n' % l, conf_matrix)

        y_error_ratio_avg.append(100 * np.mean(y_error_ratio_l))
        print('y_error_ratio_avg: ', model, test_dataset_ratio, y_error_ratio_avg)
        total_time.append((time.time() - start))
    print("total time for ", model, ": ", (time.time() - start))

print("total time: ", (time.time() - start0))