import tensorflow as tf
import numpy as np
import scipy.io as sio

data = sio.loadmat('/media/hardik/DataPart/Video1Features.mat')

features = data['features']
labels = data['frameLabels']

n = features.shape[1]

cnt = 0
for i in range(n):
    batch_X = features[0, i]
    if (batch_X.shape[0] == 0):
        continue
    print(batch_X.shape)
    cnt += 1


print(cnt)
