import matplotlib.pyplot as plt
import numpy as np


plt.figure(1)

plt.plot([-2, 0],[0, 0])
plt.plot([0, 0],[0, 1])
plt.plot([0, 2],[1, 1])
plt.xlabel("x")
plt.ylabel("heaviside(x)")

plt.figure(2)
x = np.linspace(-2, 2, 100)
y = np.tanh(x)
plt.plot(x, y)
plt.plot([-2, 2],[0, 0])
plt.plot([0, 0],[-1, 1])
plt.xlabel("x")
plt.ylabel("tanh(x)")

plt.figure(3)
x = np.linspace(-5, 5, 100)
y = np.tanh(x)
y2 = 1.0/(1.0 + np.exp(-x))
plt.plot(x, y2)
plt.plot([-10, 10],[0, 0])
plt.plot([0, 0],[-1, 1])
plt.xlabel("x")
plt.ylabel("sigmoid(x)")

plt.figure(4)
x = np.linspace(-5, 5, 100)
y = np.tanh(x)
y2 = 1.0/(1.0 + np.exp(-x))
y3 = y2 * (1 - y2)
plt.plot(x, y2, label=r'$\sigma(x)$')
plt.plot(x, y3, label=r'$\sigma(x)/dx$')
plt.plot([-5, 5],[0, 0])
plt.plot([0, 0],[-1, 1])
plt.xlabel("x")
plt.ylabel("$\sigma$(x)")
plt.legend()
plt.show()