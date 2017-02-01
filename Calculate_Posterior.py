import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import pca
from sklearn.model_selection import train_test_split
import scipy.io as sio

sess = tf.Session()

def get_hog_features():
    data = sio.loadmat('/home/hardik/Desktop/MTech_Project/Data/HOG_Feature_Data/SingleVideo/3/area9.mat')
    features = data['area_feature']
    labels = data['area_label'].flatten()
    labels = sess.run(tf.one_hot(labels, depth=4))
    # print(labels.shape)

    # Normalizing The Data
    from sklearn.preprocessing import Normalizer
    normalized_features = Normalizer().fit(features).transform(features)

    # Dimension Reduction
    pcaModel = pca.PCA(n_components=50)
    pca_features = pcaModel.fit_transform(normalized_features)

    return {'features': pca_features, 'labels': labels}

def NNModel(dimensions=[50, 25, 10], n_class=4):
    x = tf.placeholder(tf.float32, shape=[None, dimensions[0]])
    y = tf.placeholder(tf.float32, shape=[None, n_class])

    current_input = x

    # Build The classifier
    weights = []
    biases = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        w = tf.Variable(tf.random_normal(shape=[n_input, n_output]), tf.float32)
        b = tf.Variable(tf.zeros([n_output]), tf.float32)
        weights.append(w)
        biases.append(b)
        current_input = tf.nn.tanh(tf.add(tf.matmul(current_input, w), b))

    # Creating Output Layer
    n_input = int(current_input.get_shape()[1])
    w = tf.Variable(tf.random_normal(shape=[n_input, n_class]), dtype=tf.float32)
    b = tf.Variable(tf.zeros(shape=[n_class]), dtype=tf.float32)
    output = tf.add(tf.matmul(current_input, w), b)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y))
    return {'x': x, 'y': y, 'weights': weights, 'biases': biases, 'cross_entropy': cross_entropy, 'output': output}

data = get_hog_features()
features = data['features']
labels = data['labels']

"""Neural Network Parameters"""
learning_rate = 0.01

NN = NNModel()
output = NN['output']
y_pred = tf.nn.softmax(output)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(NN['y'], 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(NN['cross_entropy'])
saver = tf.train.Saver()

saver.restore(sess=sess,
              save_path='/home/hardik/Desktop/MTech_Project/Scripts/Python/MTech_Brain_Research_Python/SavedModels/nm_hog.ckpt-150')

Accuracy = sess.run(accuracy, feed_dict={NN['x']: features, NN['y']: labels})

""" Calculating Posterior for Respictive Data"""
result = sess.run(y_pred, feed_dict={NN['x']: features})
ind = np.argmax(labels, axis=1)
posterior = 0.0
for row, i in enumerate(ind):
    posterior = posterior + result[row, i]
print(posterior / features.shape[0])

# Close The Session
sess.close()
