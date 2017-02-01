"""
5 Movies Used by rajeev, converting into surf feature of each frame and classify it using SVM
"""
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib as skflow
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_validation import *
from sklearn.svm import SVC, LinearSVC, NuSVC

data = sio.loadmat("/home/hardik/Desktop/MTech_Project/MTechData/MingData/SURFFeatures.mat")
X = data['X']
y = data['y']
y2 = data['y2']

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.25)

y_train = y_train.flatten()
y_test = y_test.flatten()

print(y_train.flatten().shape)

model1 = NuSVC()
model1.fit(X_train, y_train)
print(model1.score(X_test, y_test))
