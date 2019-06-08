"""
.. versionadded:: 1.1.0
   This demo depends on new features added to contourf3d.
"""
from matplotlib import cm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

def f (theta1, theta2):
    f_theta = 0.5 * np.square (np.square (theta1) - theta2) + 0.5 * np.square (theta1 - 1)
    return f_theta

def contour_2d():
    X = np.arange(0, 2, 0.01)
    Y = np.arange(-0.5, 3, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = f (X, Y)
    CS = plt.contour(X, Y, Z, 200, linewidths = 0.5)
    
    plt.colorbar() 
    #plt.scatter(x, y, marker = 'o', c = 'b', s = 10, zorder = 10)
    plt.xlim(0, 2)
    plt.ylim(-0.5, 3)

contour_2d()

 
def g1 (theta1, theta2):
    return 2 * (theta1 * theta1 - theta2) * theta1 + (theta1 -1)

def g2 (theta1, theta2):
    return theta2 - theta1 * theta1
 
eta = 0.2
N = 500
theta1 = 0
theta2 = 2.5
f_value = f (theta1, theta2)
theta1_list, theta2_list = [], []
epsilon = 0.001

for k in range (N):
    theta1_list.append (theta1)
    theta2_list.append (theta2)

    if np.random.ranf() < 0.5:
        theta1 = theta1 - eta * g1(theta1, theta2)
    else:
        theta2 = theta2 - eta * g2(theta1, theta2)
    
    gradient =  (g1(theta1, theta2) + g2(theta1, theta2)) 
    f_value = f (theta1, theta2)
    print(k, gradient, epsilon, f_value, theta1, theta2)   

    if np.abs(gradient) < epsilon:
        break
plt.plot (theta1_list, theta2_list, linewidth = 1, linestyle ='-', marker ='*', color = 'g', markersize = 2) 
plt.scatter(theta1, theta2, color = 'r') 
plt.title ('$\eta = %0.1f,\ \epsilon=%0.4f,\ steps =%i$' % (eta, epsilon, k), color = 'r')   
plt.text(0.25, 1, '$\\theta_1=%0.3f,\ \\theta_2=%0.3f, \ f=%0.5f$' % (theta1, theta2, f_value), color = 'r')
plt.show()
