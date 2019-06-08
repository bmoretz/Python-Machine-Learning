import random

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original', '../data')
#mnist = fetch_mldata('MNIST original', '~/scikit_learn_data/mldata')
#mnist = fetch_mldata('MNIST original')

data, y = mnist.data, mnist.target
print('type of data:', type(data), ' shape of data:', data.shape)
print('type of y:',type(y), ' shape of y:', y.shape)
print('data:', data, '\ndata[0]', data[0], '\ny', y)

data, y  = shuffle(mnist.data, mnist.target, random_state=0)

def check_mnist_data (data, y, N):
    N_max = data.shape[0]
    for k in range(N):
        index = random.randint(1, N_max)
        image = data[index].reshape(28, 28)
        target = y [index]
    
        plt.subplot(5, 20, k + 1)
        plt.axis('off')
        plt.title('%i' % target, color='red')
        plt.imshow(image, cmap='Greys', interpolation = 'nearest')
    
    plt.show()

check_mnist_data (data, y, 100)

