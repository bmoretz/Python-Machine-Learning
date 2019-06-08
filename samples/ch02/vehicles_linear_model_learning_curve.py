'''
Created on Nov 26, 2017

@author: henryliu
'''
import sys

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reshape_my_data (my_data, features):
    #print ("reshape my_data: ", my_data.shape)
    X_sorted = my_data.sort_values(by=features)
    X_grouped_by_mean =  pd.DataFrame({'mean' : \
        X_sorted.groupby(features[0])[features[1]].mean()}).reset_index() 
    
    X_reshaped = X_grouped_by_mean [features[0]].values.reshape(-1,1)
    y_reshaped = X_grouped_by_mean ['mean'].values.reshape(-1,1)

    return X_reshaped, y_reshaped
    # replace=True means duplicate samples allowed
def split_my_data (my_data, test_size): 
    test_set = my_data.sample(frac=test_size, replace=False)
    train_set = my_data.sample(frac=(1 - test_size), replace=False)
    return train_set, test_set

def main(argv):
    # load pre-processed data with observed values
    data_source = "../data/vehicles_processed_mean.csv"
    fuel_economy = pd.read_csv(data_source, low_memory=False)
    #split data for training and testing
    train_set_all, test_set = split_my_data(fuel_economy, test_size=0.3)
    print ("test set size: ", len(test_set), "training set size: ", len(train_set_all))
    features = ['displ', 'mean']
    #split data with features
    test_set_x, test_set_y = reshape_my_data(test_set, features)
    
    #create poly features object with given degree
    poly_features = PolynomialFeatures(degree = 2)
    # create poly features for fixed dataset
    test_set_x_poly = poly_features.fit_transform(test_set_x)
    # create lists for plotting rmse vs training data size
    x, train_rmse, test_rmse= [], [], []
    
    for n in range (1, len (train_set_all), 1):
        x.append(n)
        train_set = train_set_all.sample (n = n, replace=False)
        train_set_x, train_set_y = reshape_my_data(train_set, features)

        train_set_x_poly = poly_features.fit_transform(train_set_x)
    
        model = linear_model.LinearRegression()
        model.fit(train_set_x_poly, train_set_y)
        
        train_set_y_pred = model.predict(train_set_x_poly)
        train_rmse.append(np.sqrt(mean_squared_error(train_set_y, train_set_y_pred)))
        
        test_set_y_pred = model.predict(test_set_x_poly)        
        test_rmse.append(np.sqrt(mean_squared_error(test_set_y, test_set_y_pred)))
    
    #plot rmse vs training data size   
    plt.plot (x, train_rmse, "r-o", label = "training set")
    plt.plot (x, test_rmse, "g-*", label = "testing set")
    plt.legend(loc='upper right', fontsize = 15)
    plt.xlabel ("# of training samples", fontsize = 15)
    plt.ylabel ("RMSE", fontsize = 15)
    plt.show()
# entry point to the main function    
if __name__ == '__main__':
    main (sys.argv)