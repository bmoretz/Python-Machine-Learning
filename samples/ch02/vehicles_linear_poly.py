'''
Created on Nov 26, 2017

@author: henryliu
'''
import sys, collections

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(argv): 

    # read in vehicles csv
    vehicles_df = pd.read_csv("../data/vehicles.csv", low_memory=False)
    
    vehicles_displ_mpg_all = vehicles_df[['displ', 'UHighway']]
    vehicles_displ_mpg = vehicles_displ_mpg_all[vehicles_displ_mpg_all.displ > 0]
    
    half = int(len (vehicles_displ_mpg) / 2)
    
    first_half = vehicles_displ_mpg [:half]
    second_half = vehicles_displ_mpg [half:]
    
    first_half_sorted = first_half.sort_values(by=['displ', 'UHighway'])
    first_half_grouped_by_median =  pd.DataFrame({'train_median' : first_half_sorted.groupby('displ')['UHighway'].median()}).reset_index() 
    
    first_half_x = first_half_grouped_by_median ['displ'].values.reshape(-1,1)
    print(type(first_half_x))
    first_half_y = first_half_grouped_by_median ['train_median'].values.reshape(-1,1)

    second_half_sorted = second_half.sort_values(by=['displ', 'UHighway'])
    second_half_grouped_by_median =  pd.DataFrame({'test_median' : second_half_sorted.groupby('displ')['UHighway'].median()}).reset_index() 
    
    second_half_x = second_half_grouped_by_median ['displ'].values.reshape(-1,1)
    second_half_y = second_half_grouped_by_median ['test_median'].values.reshape(-1,1)

    colors = ['green', 'blue', 'red', 'purple']
    lw = 1

    for count, degr in enumerate([1, 2, 4, 8]):
        # Create linear regression object
        poly_features = PolynomialFeatures(degree = degr, include_bias = False)
        first_half_x_poly = poly_features.fit_transform(first_half_x)
    
        model = linear_model.LinearRegression()
        model.fit(first_half_x_poly, first_half_y)

        # Make predictions using the testing set
        second_half_x_poly = poly_features.fit_transform(second_half_x)
        second_half_y_pred = model.predict(second_half_x_poly)
        plt.plot(second_half_x, second_half_y_pred, color=colors[count], linewidth=lw,
             label="degree %d" % degr)
        
        print("degree %d: " % degr)
        print('\tIntercept: ', model.intercept_)
        print('\tCoefficients: ', model.coef_)
        mse = mean_squared_error(second_half_y, second_half_y_pred)
        rmse = np.sqrt(mse)
        print("\tMean squared error: %.2f" % mse)
        print("\tRoot mean squared error: %.2f" % rmse)
        # Explained variance score: 1 is perfect prediction
        print('\tR-squared score: %.2f' % r2_score(second_half_y, second_half_y_pred))
        
        print ("\n\tCompare with mse and rmse for the training set:")
        first_half_y_pred = model.predict(first_half_x_poly)
        mse = mean_squared_error(first_half_y, first_half_y_pred)
        rmse = np.sqrt(mse)
        print("\tMean squared error: %.2f" % mse)
        print("\tRoot mean squared error: %.2f" % rmse)
        print('\tR-squared score: %.2f' % r2_score(first_half_y, first_half_y_pred))
    # Plot outputs
    plt.scatter(second_half_x, second_half_y,  color='gold')

    plt.xlabel ("Engine displacement (liter)")
    plt.ylabel ("Fuel economy (MPG)")
    plt.legend(loc='lower left')
    plt.show()
  
# entry point to the main function    
if __name__ == '__main__':
    main (sys.argv)