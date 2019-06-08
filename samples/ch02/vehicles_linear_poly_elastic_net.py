'''
Created on Nov 26, 2017

@author: henryliu
'''
import sys, csv

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(argv): 

    # read in vehicles csv
    vehicles_df_all = pd.read_csv("../data/vehicles.csv", low_memory=False)

    vehicles_displ_mpg_all = vehicles_df_all[['displ', 'UHighway']]
    vehicles_displ_mpg = vehicles_displ_mpg_all[vehicles_displ_mpg_all.displ > 0 ]

    #vehicles_displ_mpg = vehicles_displ_mpg.query('displ >= 1.5 and displ<= 6.8')
        
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
    for count, degree in enumerate([1, 2, 4, 8]):
        model = make_pipeline(PolynomialFeatures(degree), ElasticNet(alpha=1.0,l1_ratio=0.7))
        model.fit(first_half_x, first_half_y)
        second_half_y_pred = model.predict(second_half_x)

        plt.plot(second_half_x, second_half_y_pred, color=colors[count], linewidth=lw,
             label="degree %d" % degree)
        print("degree %d" % degree)
        mse = mean_squared_error(second_half_y, second_half_y_pred)
        rmse = np.sqrt(mse)        
        print("\tModel parameters: ", model.get_params(deep=False))
        print("\tMean squared error: %.2f" % mse,"\tRoot mean squared error: %.2f" % rmse, \
            '\tR-squared score: %.2f' % r2_score(second_half_y, second_half_y_pred))

    # Plot outputs
    plt.scatter(second_half_x, second_half_y,  color='gold')
    #plt.plot(second_half_x, second_half_y_pred, color='blue', linewidth=3)
    plt.title ("alpha=1.0", fontsize=15)
    plt.xlabel ("Engine displacement (liter)", fontsize = 15)
    plt.ylabel ("Fuel economy (MPG)", fontsize = 15)
    plt.legend(loc='upper right', fontsize = 15)
    plt.show()
  
# entry point to the main function    
if __name__ == '__main__':
    main (sys.argv)