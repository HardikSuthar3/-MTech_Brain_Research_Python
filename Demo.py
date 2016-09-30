import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing, svm, cross_validation, metrics
import math

"""Prepare The Data"""
data = sio.loadmat('/media/hardik/DataPart/4MovieSURFFeatures.mat')
X = data['FEATURES']
y_ = data['LABELS']

# print(y_[0, :])
lblBinary = preprocessing.LabelBinarizer()

# Converting into One Hot Encoding
y_ = lblBinary.fit_transform(y_)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y_, train_size=0.7)
print(y_test.shape)
