'''
Created on Nov 26, 2017

@author: henryliu
'''
import sys, csv

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import pandas as pd


def main(argv): 

    # read in vehicles csv
    # vehicles_df = pd.read_csv("data/vehicles.csv")
    vehicles_df = pd.read_csv("../data/vehicles.csv", low_memory=False)

    # pandas Series object
    vehicles_displ = vehicles_df['displ']
    print ("type of vehicles_displ: " + str(type(vehicles_displ)))
    print("First 5 rows of vehicles_displ:\n" + str(vehicles_displ.head(5)))
    print("Last 5 rows of vehicles_displ:\n" + str(vehicles_displ.tail(5)))
    print ("value_counts of vehicles_displ:\n")
    print(vehicles_displ.value_counts())
    print ("description of vehicles_displ:\n")
    print(vehicles_displ.describe())
    print ("info of vehicles_displ:\n")

    #filer by condition
    vehicles_2010 = vehicles_df [vehicles_df.year == 2010 ]
    print ("type of vehicles_2010: " + str(type(vehicles_2010)))
    print("info of vehicles_2010:\n")
    print(vehicles_2010.info)
    
    vehicles_displ.hist (bins=60, figsize = (4, 3))

    # pandas DataFrame objects
    attributes = ['displ', 'year', 'UHighway', 'UCity', 'fuelType']    
    vehicles_data_all = vehicles_df [attributes]
    print ("type of vehicles_data_all: " + str(type(vehicles_data_all)))
    
    vehicles_data = vehicles_data_all [(vehicles_data_all.fuelType == 'Regular') | \
        (vehicles_data_all.fuelType == 'Premium') | (vehicles_data_all.fuelType == 'Diesel')]
    print (vehicles_data.describe())
    
    vehicles_data.hist (bins=100, figsize = (10, 7), color='limegreen')
    vehicles_data.plot (kind="scatter", xlim= [0, 10], x = "displ", y = "UHighway", s = 2.5, c = "g", marker="s")
    vehicles_data.plot (kind="scatter", xlim= [0, 10], x = "displ", y = "UCity", s = 2.5, c = "r", marker="o")
    
    colors_palette = {'Regular': "red", 'Premium': "green", 'Diesel': 'yellow'}

    groups = list(vehicles_data.fuelType)
    colors = [colors_palette[c] for c in groups] 
    scatter_matrix ( vehicles_data [attributes], figsize = (12, 8), alpha=0.2, color=colors, diagonal='kde')
    plt.show()
    
# entry point to the main function    
if __name__ == '__main__':
    main (sys.argv)