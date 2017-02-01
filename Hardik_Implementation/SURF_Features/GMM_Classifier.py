"""
Classify mouse response image in 6 different types (Dir,NM,Plaid,SF,TF,OF) etc
"""

import scipy.io as sio
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.mixture import GMM
import sklearn.metrics as metrics

data = sio.loadmat("/home/hardik/Desktop/MTech_Project/MTechData/MingData/Feature/ClassificationFeature_2.mat")
X = data['X']
Y = data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

from sklearn.feature_selection import VarianceThreshold

sio.savemat('/home/hardik/Desktop/MTech_Project/Scripts/Mathematica/Feature.mat',
            {'trainData': X_train, 'trainLabel': y_train,
             'testData': X_test, 'testLabel': y_test})

#
# GMM_Classifier = GMM(n_components=6)
# GMM_Classifier.fit(X=X_train, y=Y)
# print(metrics.accuracy_score(GMM_Classifier.predict(X_test), y_test))
#
# print(np.unique(GMM_Classifier.predict(X_test)))
