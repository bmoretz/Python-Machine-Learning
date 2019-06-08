'''
Created on Nov 26, 2017

@author: henryliu
'''
import os, sys, collections, csv, datetime

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import pandas as pd


def main(argv): 

    # read in vehicles csv
    vehicles_df = pd.read_csv("../data/vehicles.csv", low_memory=False)
        
    vehicles_displ_mpg_2 = vehicles_df[['displ', 'UHighway']]
    vehicles_displ_mpg_processed = vehicles_displ_mpg_2
    vehicles_displ_mpg_processed = vehicles_displ_mpg_2[vehicles_displ_mpg_2.displ > 0]
    vehicles_displ_mpg_sorted = vehicles_displ_mpg_processed.sort_values(by=['displ', 'UHighway'])
    
    print (vehicles_displ_mpg_sorted.head(5))
    print (vehicles_displ_mpg_sorted.tail(5))
    print (vehicles_displ_mpg_sorted.describe())
        
    vehicles_displ_mpg_grouped = vehicles_displ_mpg_sorted.groupby('displ')['UHighway'].median()
    print(type(vehicles_displ_mpg_grouped))
    
    print (vehicles_displ_mpg_grouped.head(5))
    print (vehicles_displ_mpg_grouped.tail(5))
    print (vehicles_displ_mpg_grouped.describe())

    df_mean = pd.DataFrame({'mean' : vehicles_displ_mpg_sorted.groupby('displ')['UHighway'].mean()}).reset_index() 
    df_mean.to_csv("../data/vehicles_processed_mean.csv")
    ax = df_mean.plot (x = "displ", y = "mean", c = "r")
    #df_plot = df.plot (x = "displ", y = "median", c = "r", xlim= [8, 10], ylim= [20, 40],)
            
    df_median = pd.DataFrame({'median' : vehicles_displ_mpg_sorted.groupby('displ')['UHighway'].median()}).reset_index() 
    df_median.plot (ax=ax, x = "displ", y = "median", c = "gold")
    print (df_median.head(5))
    print (df_median.tail(5))
    print (df_median.describe())
    
    ax.set_xlabel ("Engine displacement (Liter)")
    ax.set_ylabel ("Fuel Economy (MPG)")
    plt.scatter (vehicles_displ_mpg_processed .displ, vehicles_displ_mpg_processed .UHighway, s = 2.5, c = "g", marker="s")

    plt.title('Fuel Economy vs engine displ. with pre-processed data', loc = 'center')

    plt.show()
   
# entry point to the main function    
if __name__ == '__main__':
    main (sys.argv)