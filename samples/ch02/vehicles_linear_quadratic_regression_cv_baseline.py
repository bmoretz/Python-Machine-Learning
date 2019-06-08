'''
Created on Nov 26, 2017

@author: henryliu
'''
'''
    assuming two features of (x, y), with a filter
'''

import sys, csv
import time

from sklearn import linear_model
import sklearn.datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_my_dataframe (data_source, features, selector):
    df = pd.read_csv(data_source, low_memory=False)
    #print("df type: ", type(df.iloc[2].values))
    #print("df index: ", df.iloc[2].values)
    df_selected_features = df[features]
    df_selected_features_filtered = df_selected_features[df_selected_features.displ > 0]
    return df_selected_features_filtered

    '''
    my_data is a DateFrame
    '''
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
    test_set = my_data.sample(frac=test_size, replace=True)
    train_set = my_data.sample(frac=(1 - test_size), replace=True)
    #print ("split for test set:", test_set.shape)
    #print ("split for train set:", train_set.shape)
    return train_set, test_set

    # for linear models only
def scores_and_plot (model, test_set_x, test_set_y, test_set_y_pred):  
    print('Intercept & Coefficients: ', model.intercept_, model.coef_)
    mse = mean_squared_error(test_set_y, test_set_y_pred)
    rmse = np.sqrt(mse)
    print("Mean squared error: %.2f " % mse, "Root mean squared error: %.2f" % rmse,  \
        'R-squared score: %.2f' % r2_score(test_set_y, test_set_y_pred))

    # Plot outputs
    plt.scatter(test_set_x, test_set_y,  s=12, color='red')
    plt.plot(test_set_x, test_set_y_pred, color='blue', linewidth=2)

    plt.xlabel ("Engine displacement (liter)")
    plt.ylabel ("Fuel economy (MPG)")
      
def main(argv): 
    start = time.time()
    data_source = "../data/vehicles.csv"
    features = ['displ', 'UHighway']
    #target = "UHighway"
    selector = "vehicles_displ_mpg_all.displ > 0"
    
    fuel_economy = load_my_dataframe (data_source, features, selector)
    train_set, test_set = split_my_data(fuel_economy, test_size=0.5)
    
    train_set_x, train_set_y = reshape_my_data(train_set, features)
    test_set_x, test_set_y = reshape_my_data(test_set, features)
    
    poly_features = PolynomialFeatures(degree = 2, include_bias = False)
    train_set_x_poly = poly_features.fit_transform(train_set_x)
    
    model = linear_model.LinearRegression()
    model.fit(train_set_x_poly, train_set_y)
    
    # Make predictions using the testing set
    test_set_x_poly = poly_features.fit_transform(test_set_x)
    test_set_y_pred = model.predict(test_set_x_poly)

    scores_and_plot (model, test_set_x, test_set_y, test_set_y_pred)

    print ("total time (seconds): ", (time.time() - start))
    plt.show()
# entry point to the main function    
if __name__ == '__main__':
    main (sys.argv)