import time
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np

my_data_home = '/Users/henryliu/mspc/ml_dev/ml_quantitative/data'
mnist = fetch_mldata('MNIST original', my_data_home)

data, target  = shuffle(mnist.data/255, mnist.target, random_state=0)

n = data.shape[0]
n = 20000
X, y = data[0:n], target[0:n]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=np.random.RandomState (None))

sgd = SGDClassifier(tol = 1e-3)
sgd_a = SGDClassifier(average = True, tol = 1e-3)
lr = LogisticRegression(tol = 1e-3)

dt = DecisionTreeClassifier(max_depth=10, criterion='entropy')
linear_SVC = LinearSVC(tol = 1e-3)
VC_linear_kernel = SVC(kernel='linear', tol = 1e-3)
SVC_rbf_kernel = SVC(kernel='rbf', tol = 1e-3)

individual_clfs = [('sgd', sgd), ('sgd_a', sgd_a), ('lr', lr), ('dt', dt), \
    ('linear_SVC', linear_SVC), ('SVC_linear_kernel', VC_linear_kernel), \
    ("SVC_rbf_kernel", SVC_rbf_kernel)]

voting_clf = VotingClassifier(estimators=individual_clfs, voting='hard')

for name, clf in individual_clfs:
    start = time.time()
    clf.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = str(int(fit_end - start))
    y_pred = clf.predict (X_test)
    predict_time = str(int(time.time() - fit_end))
    print(name, " score: ", accuracy_score(y_test, y_pred), 
        " fit/predict taken ", fit_time, "/", predict_time, " seconds")

start = time.time()
voting_clf.fit(X_train, y_train)
fit_end = time.time()
fit_time = str(int(fit_end - start))
y_pred = voting_clf.predict (X_test)

predict_time = str(int(time.time() - fit_end))
print("voting_clf: ", " score: ", accuracy_score(y_test, y_pred), 
    " fit/predict taken ", fit_time, "/", predict_time, " seconds")