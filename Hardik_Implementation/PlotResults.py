import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = np.loadtxt(
    '/home/hardik/Desktop/MTech_Project/Scripts/Python/MTech_Brain_Research_Python/Hardik_Implementation/Verification_Experiment'
    '/SIFT_Experiment/surf_result.txt')
print(data)
plt.hist(range(1, 10, 1), weights=data)
plt.ylim(ymax=1, ymin=0)
plt.title("")
plt.show()
