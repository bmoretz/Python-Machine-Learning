'''
Created on Nov 26, 2017

@author: henryliu
'''
import sys, csv
import time

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f, axarr = plt.subplots (2, 4)

cv_mse_scores = []
cv_rmse_scores = []
cv_r2_scores = []
    
test_mse_scores = []
test_rmse_scores = []
test_r2_scores = []
        
'''
    assuming two features of (x, y), with a filter
'''
def load_my_dataset (data_source, features, selector):
    df = pd.read_csv(data_source, low_memory=False)
    
    df_selected_features = df[features]
    df_selected_features_filtered = df_selected_features[df_selected_features.displ > 0]
    return df_selected_features_filtered

    '''
    my_data is a DateFrame
    '''
def reshape_my_data (my_data, features):
    X_sorted = my_data.sort_values(by=features)
    X_grouped_by_mean =  pd.DataFrame({'mean' : \
        X_sorted.groupby(features[0])[features[1]].mean()}).reset_index() 
    
    X_reshaped = X_grouped_by_mean [features[0]].values.reshape(-1,1)
    y_reshaped = X_grouped_by_mean ['mean'].values.reshape(-1,1)

    return X_reshaped, y_reshaped

### begin utility functions for CV
def split_kfolds (df, k):
    k = k
    sub_df = []
    for i in range(k):
        sub_df.append (pd.DataFrame(columns=df.columns))
    
    for i in range (0, len(df), k):
        for j in range (k):
            x = i + j
            if (x < len(df)):
                #print("add:", i,j,x,"\n", df.iloc[x].values, " type: ", type(df.iloc[x]))
                sub_df[j]= sub_df[j].append (df.iloc[x], ignore_index=True)

    return sub_df 

# concat k - 2 frames from k_1 frame 
def make_train_validate_frames (frames, k, k_validate):
    validate_frame = frames [k_validate]
    train_frame = pd.DataFrame()
    for i in range (k - 1):
        if (i != k_validate):
            train_frame = pd.concat ([train_frame, frames[i]])
      
    return train_frame.reset_index(), validate_frame  

def cross_validate (frames, k, features):  
    test_frame = frames [k - 1]
    for i in range (k - 1): 
        train_frame, validate_frame = make_train_validate_frames (frames, k, i) 
        process (train_frame, validate_frame, features, i, 0) 
        process (train_frame, test_frame, features, i, 1) 
        
def process (train_set, test_set, features, i, type):   
    train_set_x, train_set_y = reshape_my_data(train_set, features)
    test_set_x, test_set_y = reshape_my_data(test_set, features)
    
    poly_features = PolynomialFeatures(degree = 2, include_bias = False)
    train_set_x_poly = poly_features.fit_transform(train_set_x)
    
    model = linear_model.LinearRegression()
    model.fit(train_set_x_poly, train_set_y)
    
    # Make predictions using the testing set
    test_set_x_poly = poly_features.fit_transform(test_set_x)
    test_set_y_pred = model.predict(test_set_x_poly)

    scores_and_plot (model, test_set_x, test_set_y, test_set_y_pred, i, type)
    #print (test_frame)

### end utility functions for CV
    # replace=True means duplicate samples allowed
def split_my_data (my_data, test_size): 
    test_set = my_data.sample(frac=test_size, replace=True)
    train_set = my_data.sample(frac=(1 - test_size), replace=True)
    return train_set, test_set

    # for linear models only
def scores_and_plot (model, test_set_x, test_set_y, test_set_y_pred, i, type):
      
    print('Intercept & Coefficients: ', model.intercept_, model.coef_)
    mse = mean_squared_error(test_set_y, test_set_y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_set_y, test_set_y_pred)
    if (type == 0):
        cv_mse_scores.append (mse)
        cv_rmse_scores.append (rmse)
        cv_r2_scores.append (r2)
    else:
        test_mse_scores.append (mse)
        test_rmse_scores.append (rmse)
        test_r2_scores.append (r2)
    print("Mean squared error: %.2f " % mse, "Root mean squared error: %.2f" % rmse,  \
        'R-squared score: %.2f' % r2)

    # Plot outputs
    axarr [type, i].scatter(test_set_x, test_set_y,  s=12, color='red')
    axarr [type, i].plot(test_set_x, test_set_y_pred, color='blue', linewidth=2)

    if (type == 0 and i == 2):
        axarr [type, i].set_title ("CV runs: Fuel economy (MPG) versus Engine displacement (liters)") 
    if (type == 1 and i == 2):
        axarr [type, i].set_title ("Test runs: Fuel economy (MPG) versus Engine displacement (liters)")
    
def print_scores():
    print ("CV Scores:") 
    np_mse = np.array(cv_mse_scores)
    np_rmse = np.array(cv_rmse_scores)
    np_r2 = np.array(cv_r2_scores)
    print ("\tCV MSE Scores: ", cv_mse_scores) 
    print ("\tCV RMSE Scores:", cv_rmse_scores) 
    print ("\tCV R2 Scores:", cv_r2_scores) 
    print ("\tCV MSE Scores (mean/std): ", np_mse.mean(), "/", np_mse.std()) 
    print ("\tCV RMSE Scores (mean/std):",  np_rmse.mean(), "/", np_rmse.std()) 
    print ("\tCV R2 Scores: (mean/std)",  np_r2.mean(), "/", np_r2.std()) 
    
    print ("Test Scores:") 
    np_mse = np.array(test_mse_scores)
    np_rmse = np.array(test_rmse_scores)
    np_r2 = np.array(test_r2_scores)
    print ("\tTest MSE Scores:",  test_mse_scores) 
    print ("\tTest RMSE Scores:",  test_rmse_scores) 
    print ("\tTest R2 Scores:",  test_r2_scores)
    print ("\tTest MSE Scores (mean/std): ", np_mse.mean(), "/", np_mse.std()) 
    print ("\tTest RMSE Scores (mean/std):",  np_rmse.mean(), "/", np_rmse.std()) 
    print ("\tTest R2 Scores: (mean/std)",  np_r2.mean(), "/", np_r2.std())  
              
def main(argv): 
    start = time.time()
    data_source = "../data/vehicles.csv"
    features = ['displ', 'UHighway']
    target = "UHighway"
    selector = "vehicles_displ_mpg_all.displ > 0"
    
    fuel_economy = load_my_dataset (data_source, features, selector)
    #train_set, test_set = split_my_data(fuel_economy, test_size=0.5)
    #df = pd.DataFrame(np.random.randn(100,2))
    k = 5
    frames = split_kfolds (fuel_economy, k)

    cross_validate (frames, k, features)
    print_scores()
    print ("total time (seconds): ", (time.time() - start))
    plt.show()
# entry point to the main function    
if __name__ == '__main__':
    main (sys.argv)