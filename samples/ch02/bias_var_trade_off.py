"""bias_variance_trade_off worker
Author: Henry H. Liu
"""
print(__doc__)

import csv

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


#DataFrame to hold predicted values in a y-matrix (l, n) and averaged predictions
y_ln = pd.DataFrame()
y_mn = pd.DataFrame()

def plot_3d (X, Y, Z): 
    X, Y = np.meshgrid(X, Y) 
    #Z1 = np.sin(X) 
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,linewidth=0, antialiased=False)
    #surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,linewidth=0, antialiased=False)   
    # Customize the z axis.
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel ('x')
    ax.set_ylabel ('m')
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
    
# initializes N_data_points, L_datasets and noise_coefficient
def init (N, L, noise):
    global N_data_points, L_datasets, noise_coefficient
    global n_train, x_min, x_max, num_of_subplots
    N_data_points, L_datasets, noise_coefficient = N, L, noise
    train_ratio = 0.5
    n_train = int (train_ratio * N_data_points)
    x_min, x_max = 0, 2 * np.pi

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

def plot_original(x_sorted, degree): 

    axarr[get_index_i(degree), get_index_j(degree)].plot(x_sorted, f(x_sorted), color='black', linewidth=2, label="sine function")
    axarr[get_index_i(degree), get_index_j(degree)].scatter(x_sorted, f(x_sorted), color='black', s=5)
    
def plot_training_set (x_sorted, train_x, train_y, l, degree, color): 
    global axarr
    if l ==  0: # legend should be displayed only once       
        #print ("m, i, j = ", degree, get_index_i(degree), get_index_j(degree))
        axarr[get_index_i(degree), get_index_j(degree)].plot (train_x, train_y, color = color, lw=1, label = "training data")
    else:
        axarr[get_index_i(degree), get_index_j(degree)].plot (train_x, train_y, color = color, lw=1)
     
def plot_predicted (x_sorted, test_x, test_predicted, l, degree, color):
    if l == 0: # legend should be displayed only once
        axarr[get_index_i(degree), get_index_j(degree)].plot (test_x, test_predicted, color = color,lw=1, label = "predicted: m=%d" % degree)
    else:
        axarr[get_index_i(degree), get_index_j(degree)].plot (test_x, test_predicted, color = color,lw=1)

def train_and_predict (X_train, train_y, X_test, degree, alpha):
        # train model
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha = alpha))
        model.fit(X_train, train_y)
        # predict
        test_predicted = model.predict(X_test)
        return test_predicted

def fit_it (degree):
    train_y, test_x, X_train, X_test = get_input ()
    test_predicted = train_and_predict (X_train, train_y, X_test, degree)
    return test_predicted, f(test_x)

def update_y_ln (l, test_predicted):
    global y_ln
    y_ln = y_ln.append(pd.Series(test_predicted.ravel()), ignore_index=True)
    
def compute_bias_variance (h_n_series): 
    global y_mn
    #print(type(y_ln.mean(axis=0).values), y_ln.mean(axis=0).values)   
    y_n_bar_series = y_ln.mean(axis=0)
    y_mn = y_mn.append(y_n_bar_series, ignore_index=True)
    #pd.to_csv("../data/y_n_bar.csv", )
    bias_2 = mean_squared_error (y_n_bar_series, h_n_series)
    y_ln_deviation_2 = np.square(y_ln - y_n_bar_series)
    y_n_mean_by_row = y_ln_deviation_2.mean(axis=0)
    variance = np.mean(y_n_mean_by_row)
    print("bias_2: ", bias_2, " variance: ", variance, " total: ", (bias_2 + variance))
    return bias_2, variance
    
def worker (degree, alpha):

    for l in range (0, L_datasets):  
        x_sorted, train_x, train_y, test_x, test_y, X_train, X_test = get_input (l)
        test_predicted = train_and_predict (X_train, train_y, X_test, degree, alpha)
        if l == 0: 
            plot_original(x_sorted, degree) 
            h_n_series = pd.Series (f(train_x))
        if (l % 10 == 0):
            plot_training_set (x_sorted, train_x, train_y, l, degree, "gold")
            plot_predicted (x_sorted, test_x, test_predicted, l, degree, "red")
        update_y_ln (l, test_predicted)
        axarr[get_index_i(degree), get_index_j(degree)].legend (loc='upper right', frameon=False)
        axarr[get_index_i(degree), get_index_j(degree)].set_xlabel ("x")
    bias_2, variance = compute_bias_variance (h_n_series)

    return bias_2, variance

def get_index_i (m):
    i = int ((m - 1) / 2)
    return i

def get_index_j (m):
    j = (m - 1) % 2
    return j

def plot_bias_variance (M, poly_degree, bias_2, variance): 
    plt.xlim(0, M + 1)
    plt.ylim(0, 0.65)
    axarr[int(M / 2), int(M % 2)].plot (poly_degree, bias_2, color='green', label = r'$bias^2$')
    axarr[int(M / 2), int(M % 2)].plot(poly_degree, variance, color='red', label = "variance")
    axarr[int(M / 2),int(M % 2)].plot(poly_degree, total, color='blue', label = r'$bias^2 + variance$')
    plt.xlabel ("polynomial degree (m)", fontsize = 10)
    plt.ylabel ("Errors", fontsize = 10)
    plt.legend(loc='upper right', fontsize = 10, frameon=False)
    fig.text(0.2, 0.89, r'$Test\/\/Configuration: N=%s, L=%s, M=%s, noise=%s, \alpha=%s$' %(N_data_points, \
        L_datasets, M, noise_coefficient, alpha), fontsize=10, fontweight='bold')

def driver (N_data_points, L_datasets, noise_coefficient, alpha):

    init (N_data_points, L_datasets, noise_coefficient)
    
    for m in range (1, M + 1, 1):
        poly_degree.append (m)
        b2, var = worker (m, alpha)
        bias_2.append (b2)
        variance.append (var)
        total.append (b2 + var)
    
    print (bias_2, "\n", variance)
    plot_bias_variance (M, poly_degree, bias_2, variance)

    plot_3d(test_x,  np.arange(M), y_mn.values)

#Test configuration params:
M, N_data_points, L_datasets, noise_coefficient, alpha = 11, 200, 100, 0.3, 0.0001
fig, axarr = plt.subplots (int(M / 2) + 1, 2)

poly_degree, bias_2, variance, total = [], [], [], []
driver (N_data_points, L_datasets, noise_coefficient,alpha)