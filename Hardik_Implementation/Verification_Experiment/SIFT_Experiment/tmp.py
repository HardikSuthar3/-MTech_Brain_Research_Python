import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('/home/hardik/Desktop/MTech_Project'
                  '/Scripts/Python/MTech_Brain_Research_Python'
                  '/Hardik_Implementation/Verification_Experiment'
                  '/surf_result.txt', dtype=np.float32)

avg = np.min(data)
data = data - avg

X = [i for i in range(1, 10)]
X = np.array(range(1,10))
print(X.shape)
plt.bar(X, data)
plt.show()
