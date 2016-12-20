import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import pca
from sklearn.model_selection import train_test_split
import scipy.io as sio

sess = tf.Session()

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

def get_hog_features():
    data = sio.loadmat('/home/hardik/Desktop/MTech_Project/Data/HOG_Feature_Data/natural_movies_hog_features.mat')
    features = data['hog_features']
    labels = data['hog_labels'].flatten()
    labels = sess.run(tf.one_hot(labels, depth=4))
    print(labels.shape)

    # Normalizing The Data
    from sklearn.preprocessing import Normalizer
    normalized_features = Normalizer().fit(features).transform(features)

    # Dimension Reduction
    pcaModel = pca.PCA(n_components=50)
    pca_features = pcaModel.fit_transform(normalized_features)

    return {'features': pca_features, 'labels': labels}

data = get_hog_features()
x_train, x_test, y_train, y_test = train_test_split(data['features'], data['labels'], train_size=0.8)

def NextBatch(batchSize=16):
    data = x_train[NextBatch.batchIndex:NextBatch.batchIndex + batchSize, :], \
           y_train[NextBatch.batchIndex:NextBatch.batchIndex + batchSize, :]
    NextBatch.batchIndex += batchSize
    return data
NextBatch.batchIndex = 0

"""Neural Network Parameters"""
learning_rate = 0.01
training_epochs = 300
display_step = 100
batch_size = 16

NN = NNModel()
output = NN['output']
y_pred = tf.nn.softmax(output)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(NN['y'], 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(NN['cross_entropy'])
sess.run(tf.global_variables_initializer())

prevCost = 0.0
for ep_i in range(training_epochs):
    total_batch = int(x_train.shape[0] / batch_size)
    for batch_i in range(2):
        bx, by = NextBatch(batchSize=batch_size)
        print(type(bx))
        sess.run(optimizer, feed_dict={NN['x']: bx, NN['y']: by})
    training_cost = sess.run(NN['cross_entropy'], feed_dict={NN['x']: x_train, NN['y']: y_train})
    if np.abs(prevCost - training_cost) < 0.00001:
        break

Accuracy = sess.run(accuracy, feed_dict={NN['x']: x_test, NN['y']: y_test})
print("Accuracy %f" % Accuracy)
sess.close()
