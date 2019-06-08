import matplotlib.pyplot as plt
import numpy as np


plt.figure(1)

x = np.linspace(0, 2, 100)
y = x
plt.plot(x, y, c='b')
plt.plot([-2, 2],[0, 0])
plt.plot([-2, 0],[0, 0], c='b')
plt.plot([0, 0],[-1, 2])
plt.xlabel("x")
plt.ylabel("ReLU")


plt.figure(2)
x = np.linspace(-2, 2, 100)
y = np.tanh(x)
plt.plot(x, y)
plt.plot([-2, 2],[0, 0])
plt.plot([0, 0],[-1, 1])
plt.xlabel("x")
plt.ylabel("tanh(x)")


plt.show()