import random, time

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np

#my_data_home = '/Users/henryliu/Documents/ml_dev/ml_quantitative/data'
#my_data_home = '~/scikit_learn_data/mldata'
#mnist = fetch_mldata('MNIST original', data_home=my_data_home)
mnist = fetch_mldata('MNIST original', '../data')

def plot_scores (x, data, n_rows, n_columns, index, xlabel, ylabel): 
    #plt.ylim(0, 100)
    plt.subplot(n_rows, n_columns, index)
    plt.plot(x, data)
    plt.scatter(x, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.legend(loc = "upper right")
    if (index == 1):
        plt.title('Total # of samples: %s' % (n))
    #plt.legend(frameon=False)
    plt.grid(True) 

def plot_histogram (y_scores):
    mean = int(np.average(y_scores))
    min = int(np.min(y_scores))
    max = int(np.max(y_scores))
    
    plt.hist(y_scores, bins='auto')
    plt.xlabel('Confidence scores')
    plt.ylabel('Count')
    plt.title(' SGD_A classifier confidence scores with min =' \
        + str(min) + ", mean = " + str(mean) + " & max = " + str(max))
    plt.grid(True)
    
baseline = True
#baseline = False
start0 = time.time()
data, target  = shuffle(mnist.data/255, mnist.target, random_state=0)

n = data.shape[0]
#n = 20000
X, y = data[0:n], target[0:n]

clf = SGDClassifier(average = True, tol = 1e-3)

test_dataset_ratio = [0.1, 0.25, 0.5, 0.75, 0.9]

rng = np.random.RandomState (None)
L = 6

y_accuracy_avg, precision_avg, recall_avg, f1_avg = [], [], [], []

for split_ratio in test_dataset_ratio:
    y_accuracy_l, precision_l, recall_l, f1_l = [], [], [], []
    for l in range (L):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, random_state=rng)

        y_train_9 = (y_train == 9)
        y_test_9 = (y_test == 9)

        clf.fit(X_train, y_train_9) 
        if baseline:
            y_pred = clf.predict(X_test)
        else:
            y_scores = clf.decision_function(X_test)
            mean_score = np.mean(y_scores)
            if (l == 0 and split_ratio == 0.5):
                plot_histogram (y_scores)
            #y_pred = (y_scores > mean_score)
            y_pred = (y_scores > 10)
        y_accuracy_l.append (np.mean(y_pred == y_test_9))
        precision_l.append (precision_score(y_test_9, y_pred))
        recall_l.append (recall_score(y_test_9, y_pred))
        f1_l.append (f1_score(y_test_9, y_pred))

    y_accuracy_avg.append(100 * np.mean(y_accuracy_l))
    precision_avg.append(100 * np.mean(precision_l))
    recall_avg.append(100 * np.mean(recall_l))
    f1_avg.append(100 * np.mean(f1_l))
    print(1 - split_ratio, y_accuracy_avg, precision_avg, recall_avg, f1_avg)

n_rows = 3
n_columns = 2
index = 1
x_axis_data = 1. - np.array(test_dataset_ratio)
plot_scores (x_axis_data, y_accuracy_avg, n_rows, n_columns, index, "Training set ratio", "Accuracy(%)")
plot_scores (x_axis_data, precision_avg, n_rows, n_columns, index + 1, "Training set ratio", "Precision score(%)")
plot_scores (x_axis_data, recall_avg, n_rows, n_columns, index + 2, "Training set ratio", "Recall score(%)")
plot_scores (x_axis_data, f1_avg, n_rows, n_columns, index + 3, "Training set ratio", "F1 score(%)")
    
print("total time: ", (time.time() - start0))
plt.show()