import numpy as np
from sklearn.model_selection import StratifiedKFold

X = np.ones(10)
y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

print('X: ', X)
print('y: ', y)

skf = StratifiedKFold(n_splits = 3)
for train_index, test_index in skf.split(X, y):
    print('train_index, test_index:', train_index, test_index)