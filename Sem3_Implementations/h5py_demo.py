import numpy as np
import h5py
import scipy.io as sio

# Load using H5py
data = h5py.File("/home/hardik/Desktop/MTech_Project/MTechData/MingData/SURF2.mat", 'r')
print(list(data.keys()))

X = data.get('X').value.transpose()
y = data['y'].value
y2 = data['y2'].value


print(z)

# print(list(y.dims))

#
# data = sio.loadmat("/home/hardik/Desktop/MTech_Project/MTechData/MingData/SURFFeatures.mat")
# X = data['X']
# y = data['y']
# y2 = data['y2']
# print(X.shape)
