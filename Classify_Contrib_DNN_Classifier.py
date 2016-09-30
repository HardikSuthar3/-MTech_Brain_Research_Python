import scipy.io as sio
import numpy as np
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from sklearn import preprocessing
data = sio.loadmat("/home/hardik/Desktop/MTech_Project/MTechData/MingData/Feature/ClassificationFeature.mat")
X = data['X']
Y = data['Y']
Y = np.ndarray.astype(Y, np.int64)
print(Y.shape)
# X=preprocessing.scale(X)
print(np.mean(X,axis=1))


# X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.6)
#
# classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10])
# classifier.fit(X_train, y_train, steps=200)
#
# accuracy = classifier.evaluate(x=X_test, y=y_test)["accuracy"]
# print("Accuracy: {0:f}", format(accuracy))
