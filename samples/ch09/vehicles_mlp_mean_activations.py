'''
Created on Nov 26, 2017

@author: henryliu
'''
import sys, csv

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(argv): 

    # read in vehicles csv
    vehicles_df = pd.read_csv("../data/vehicles.csv", low_memory=False)
    
    vehicles_displ_mpg_all = vehicles_df[['displ', 'UHighway']]
    vehicles_displ_mpg = vehicles_displ_mpg_all[vehicles_displ_mpg_all.displ > 0]
    
    half = int(len (vehicles_displ_mpg) / 2)
    
    # create the training set with the first half of data
    first_half = vehicles_displ_mpg [:half]
    second_half = vehicles_displ_mpg [half:]
    
    first_half_sorted = first_half.sort_values(by=['displ', 'UHighway'])
    first_half_grouped_by_mean =  pd.DataFrame({'train_mean' : \
        first_half_sorted.groupby('displ')['UHighway'].mean()}).reset_index() 
    
    first_half_x = first_half_grouped_by_mean ['displ'].values.reshape(-1,1)
    print(type(first_half_x))
    print(first_half_x.shape)
    first_half_y = first_half_grouped_by_mean ['train_mean'].values.reshape(-1,1)
    print(first_half_y.shape)
        
    #ax = first_half_grouped_by_median.plot (x = "displ", y = "train_median", c = "b")
    #plt.show ()
    
    second_half_sorted = second_half.sort_values(by=['displ', 'UHighway'])
    second_half_grouped_by_mean =  pd.DataFrame({'test_mean' : \
        second_half_sorted.groupby('displ')['UHighway'].mean()}).reset_index() 
    
    second_half_x = second_half_grouped_by_mean ['displ'].values.reshape(-1,1)
    second_half_y = second_half_grouped_by_mean ['test_mean'].values.reshape(-1,1)
    #second_half_grouped_by_median.plot (ax=ax, x = "displ", y = "test_median", c = "gold")
    #plt.show ()
    
    # Create linear regression object
    activations = ['identity', 'logistic', 'tanh', 'relu']
    for activation in activations:
        regr = MLPRegressor(hidden_layer_sizes=(100,), activation=activation, max_iter=200, 
            alpha=1e-4, solver='lbfgs', verbose=10, tol=1e-4, random_state=1,learning_rate_init=.1)

    # Train the model using the training sets
        regr.fit(first_half_x, first_half_y.ravel())
    
        # Make predictions using the testing set
        second_half_y_pred = regr.predict(second_half_x)
    
        print("\tModel parameters: ", regr.get_params(deep=False))
    
        # mean squared error
        mse = mean_squared_error(second_half_y, second_half_y_pred)
        rmse = np.sqrt(mse)
        print("Mean squared error: %.2f" % mse)
        print("Root mean squared error: %.2f" % rmse)
        # Explained variance score: 1 is perfect prediction
        print('R-squared score: %.2f' % r2_score(second_half_y, second_half_y_pred))

        plt.plot(second_half_x, second_half_y_pred, color='blue', linewidth=2, label="activation function = %s" %activation)
        plt.scatter(first_half_x, first_half_y,  color='green', label = "training set")
        plt.scatter(second_half_x, second_half_y,  color='red', label = "testing set")
        plt.xlabel ("Engine displacement (liter)")
        plt.ylabel ("Fuel economy (MPG)")
        plt.legend(loc='upper right')
        plt.show()
  
# entry point to the main function    
if __name__ == '__main__':
    main (sys.argv)