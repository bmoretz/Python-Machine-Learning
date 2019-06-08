import time
from sklearn.datasets import fetch_mldata
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np

my_data_home = '/Users/henryliu/mspc/ml_dev/ml_quantitative/data'
mnist = fetch_mldata('MNIST original', data_home=my_data_home)

start0 = time.time()
data, target  = shuffle(mnist.data/255, mnist.target, random_state=0)

n = data.shape[0]
n = 20000
X, y = data[0:n], target[0:n]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=np.random.RandomState (None))
pasting_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, bootstrap=False, n_jobs=-1)
#pasting_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, max_samples=1000, bootstrap=False, n_jobs=-1)

start = time.time()
pasting_clf.fit(X_train, y_train)
fit_end = time.time()
fit_time = str(int(fit_end  - start))
y_pred = pasting_clf.predict (X_test)
predict_time = str(int(time.time() - fit_end))
print("pasting_clf: ", " score: ", accuracy_score(y_test, y_pred), 
    " fit/predict taken ", fit_time, "/", predict_time, " seconds")

dt = DecisionTreeClassifier(max_depth=10, criterion='entropy')
start = time.time()
dt.fit(X_train, y_train)
fit_end = time.time()
fit_time = str(int(fit_end  - start))
y_pred = dt.predict (X_test)
predict_time = str(int(time.time() - fit_end))
print("single dt: ", " score: ", accuracy_score(y_test, y_pred), 
    " fit/predict taken ", fit_time, "/", predict_time, " seconds")