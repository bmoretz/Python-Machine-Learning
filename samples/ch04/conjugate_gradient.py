"""
.. versionadded:: 1.1.0
   This demo depends on new features added to contourf3d.
"""
from matplotlib import cm
import numdifftools as nd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d
import numpy as np

def f0 (theta1, theta2):
    f_theta = 0.5 * np.square (np.square (theta1) - theta2) + 0.5 * np.square (theta1 - 1)
    return f_theta

def f (theta):
    return 0.5 * np.square (np.square (theta[0]) - theta[1]) + 0.5 * np.square (theta[0] - 1)


def contour_2d():
    X = np.arange(0, 2, 0.01)
    Y = np.arange(-0.5, 3, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = f0 (X, Y)
    CS = plt.contour(X, Y, Z, 200, linewidths = 0.5)
    
    plt.colorbar() 
    plt.xlim(0, 2)
    plt.ylim(-0.5, 3)

contour_2d()

N = 500

epsilon = 0.001
theta = [0, 0.0]
jacobian = nd.Jacobian(f) (theta)
hessian = nd.Hessian(f) (theta)
g = -jacobian

k = 0
theta1_list, theta2_list = [], []
while True:
    theta1_list.append (theta[0])
    theta2_list.append (theta[1])

    jacobian = nd.Jacobian(f) (theta)
    hessian = nd.Hessian(f) (theta)
    eta = -np.dot(jacobian, np.transpose (g)) / np.dot(np.dot(g, hessian), np.transpose (g))
    print (k, eta, eta * g) 
    theta = theta + eta * g
    dp = np.dot (g, np.transpose(g))

    k = k +1
    theta = theta.ravel()
    if (dp < epsilon * epsilon) or k > N:
        break
    jacobian_new = nd.Jacobian(f) (theta)
    beta = np.dot(jacobian_new, np.transpose (jacobian_new)) / np.dot(jacobian, np.transpose (jacobian))
    g = -jacobian_new + beta * g

f_value = f(theta)
plt.plot (theta1_list, theta2_list, linewidth = 1, linestyle ='-', marker ='*', color = 'r', markersize = 2)  
plt.scatter(theta[0], theta[1], color = 'g')  
plt.title ('$\epsilon=%0.4f,\ steps =%i$' % (epsilon, k), color = 'r')   
plt.text(0.25, 2, '$\\theta_1=%0.3f,\ \\theta_2=%0.3f, \ f=%0.5f$' % (theta[0], theta[1], f_value), color = 'r')

plt.show()
