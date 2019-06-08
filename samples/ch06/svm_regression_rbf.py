"""bias_variance_trade_off worker
Author: Henry H. Liu
"""
print(__doc__)
import time
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#DataFrame to hold predicted values in a y-matrix (l, n) and averaged predictions
y_ln = pd.DataFrame()
y_mn = pd.DataFrame()

# initializes N_data_points, L_datasets and noise_coefficient
def init (N, L, noise):
    global N_data_points, L_datasets, noise_coefficient
    global n_train, x_min, x_max, num_of_subplots
    N_data_points, L_datasets, noise_coefficient = N, L, noise
    train_ratio = 0.5
    n_train = int (train_ratio * N_data_points)
    x_min, x_max = 0, 2.0 * np.pi

# exact function to fit
def f(x):
    return np.sin(x)

#create input data for iteration l
def get_input (l):
    global test_x, test_y, X_test

    #generate data points for splitting into train/test set
    x_sorted = np.linspace(x_min, x_max, N_data_points) 
    
    #split input
    x_input = x_sorted.copy ()
    rng = np.random.RandomState(None) # random seed if None
    rng.shuffle(x_input)
    train_x = np.sort(x_input[:n_train])
    #test_x = np.sort (x_input[n_train:])
    #test_y = f (test_x)

    #prepare input matrix required for fitting and predicting
    X_train = train_x[:, np.newaxis]
    
    if l == 0:
        test_x = np.sort (x_input[n_train:])
        test_y = f (test_x)
        X_test = test_x[:, np.newaxis]

    #add noise to training data 
    train_noise = noise_coefficient * 2 * (0.5 - rng.rand(X_train.shape[0]))
    train_y = f(train_x) + train_noise

    return x_sorted, train_x, train_y, test_x, test_y, X_train, X_test

def plot_original(x_sorted, power): 
    axarr[get_index_i(power), get_index_j(power)].plot(x_sorted, f(x_sorted), color='black', linewidth=1, label="sine function")
     
def plot_predicted (x_sorted, test_x, test_predicted, l, power, color):
    gamma_value = 10 ** (power - 3)
    if l == 0:
        axarr[get_index_i(power), get_index_j(power)].plot (test_x, test_predicted, color = color,lw=1, \
        label = r'$predicted: \gamma=%2.2f"$' % gamma_value)
    else:
            axarr[get_index_i(power), get_index_j(power)].plot (test_x, test_predicted, color = color,lw=1)
def train_and_predict (X_train, train_y, X_test, power):
    # train model

    #model = SVR (kernel="poly", power=power, C=10, epsilon=0.01)
    model = SVR (kernel="rbf", gamma=10 ** (power- 3))
    #model = SVR (kernel="rbf", power=10./(power * power))
    model.fit(X_train, train_y)
    # predict
    test_predicted = model.predict(X_test)
    return test_predicted

def fit_it (power):
    train_y, test_x, X_train, X_test = get_input ()
    test_predicted = train_and_predict (X_train, train_y, X_test, power)
    return test_predicted, f(test_x)

def update_y_ln (l, test_predicted):
    global y_ln
    y_ln = y_ln.append(pd.Series(test_predicted.ravel()), ignore_index=True)

def compute_bias_variance (h_n_series): 
    global y_mn
 
    y_n_bar_series = y_ln.mean(axis=0)
    y_mn = y_mn.append(y_n_bar_series, ignore_index=True)

    bias_2 = mean_squared_error (y_n_bar_series, h_n_series)
    y_ln_deviation_2 = np.square(y_ln - y_n_bar_series)
    y_n_mean_by_row = y_ln_deviation_2.mean(axis=0)
    variance = np.mean(y_n_mean_by_row)
    print("bias_2: ", bias_2, " variance: ", variance, " total: ", (bias_2 + variance))
    return bias_2, variance

def worker (power):
    colors = ['r', 'g', 'b', 'orange']
    for l in range (0, L_datasets):  
        x_sorted, train_x, train_y, test_x, test_y, X_train, X_test = get_input (l)
        test_predicted = train_and_predict (X_train, train_y, X_test, power)
        if l == 0: 
            plot_original(x_sorted, power) 
            h_n_series = pd.Series (f(train_x))
        #if (l % 2 == 20 and l > 0):
        if (l > -1):
            plot_predicted (x_sorted, test_x, test_predicted, l, power, colors[l % 4])
        update_y_ln (l, test_predicted)
        axarr[get_index_i(power), get_index_j(power)].legend (loc='upper right', frameon=False)
        axarr[get_index_i(power), get_index_j(power)].set_xlabel ("x")
    bias_2, variance = compute_bias_variance (h_n_series)

    return bias_2, variance

# move the first subplot over
def get_index_i (m):
    i = int ((m - 1) / 2)
    return i

def get_index_j (m):
    j = (m - 1) % 2
    return j

def plot_bias_variance (M, power, bias_2, variance): 
    #plt.xlim(0, M + 1)
    #plt.ylim(0, 0.25)
    i = int(M / 2) 
    j = int(M % 2)

    axarr[i, j].plot (power, bias_2, color='green', label = r'$bias^2$')
    axarr[i, j].plot(power, variance, color='red', label = "variance")
    axarr[i, j].plot(power, total, color='blue', label = r'$bias^2 + variance$')
    plt.xlabel (r'$log_{10}(\gamma)$', fontsize = 10)
    plt.ylabel ("Errors", fontsize = 10)
    plt.legend(loc='best', fontsize = 10, frameon=False)
    
    fig.text(0.2, 0.89, r'$Test\/\/Configuration: N=%s, L=%s, M=%s, noise=%s$' %(N_data_points, \
        L_datasets, M, noise_coefficient), fontsize=10, fontweight='bold')
    
def driver (N_data_points, L_datasets, noise_coefficient):

    init (N_data_points, L_datasets, noise_coefficient)
    
    for m in range (1, M + 1, 1):
        power.append (np.log10(10**(m-3)))
        b2, var = worker (m)
        bias_2.append (b2)
        variance.append (var)
        total.append (b2 + var)
    
    print (bias_2, "\n", variance)
    plot_bias_variance (M, power, bias_2, variance)

    #plot_3d(test_x,  np.arange(M), y_mn.values)

#Test configuration params:
M, N_data_points, L_datasets, noise_coefficient = 5, 500, 50, 0.0
fig, axarr = plt.subplots (int((M - 1) / 2) + 1, 2)

power, bias_2, variance, total = [], [], [], []
driver (N_data_points, L_datasets, noise_coefficient)
plt.show()