import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn import preprocessing

sess = tf.InteractiveSession()

X = np.random.randn(5, 4)
Y = np.array([0, 1, 2, 3, 0])
Y = tf.one_hot(Y, depth=4).eval()
ind = np.argmax(Y, axis=1)
print(X)
for row, i in enumerate(ind):
    print(X[row, i])
