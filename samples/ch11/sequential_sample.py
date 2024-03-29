from keras.layers import Dense, Activation
from keras.models import Sequential


model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

