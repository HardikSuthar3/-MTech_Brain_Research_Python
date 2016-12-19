import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn import preprocessing, svm, cross_validation, metrics
import math


"""Prepare The Data"""
data = sio.loadmat('/media/hardik/DataPart/4MovieSURFFeatures.mat')
X = data['FEATURES']
y_ = data['LABELS']

# Converting into One Hot Encoding
lblBinary = preprocessing.LabelBinarizer()
y_ = lblBinary.fit_transform(y_)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y_, train_size=0.7)

"""
Train Data : 12259
factors(batch): 1 13 23 41 299 533 943 12259
"""


def NextBatch(batchSize=41):
    data = X_train[NextBatch.batchIndex:NextBatch.batchIndex + batchSize, :], \
           y_train[
           NextBatch.batchIndex:NextBatch.batchIndex + batchSize,
           :]
    NextBatch.batchIndex += batchSize
    return data


NextBatch.batchIndex = 0

"""End Of Data"""

"""Create Model"""

# svmModel = svm.SVC()  # kernel='poly',degree=5
#
# print(y_train.transpose().shape)
# svmModel.fit(X_train, y_train.ravel())
#
# y = svmModel.predict(X_test)

learning_rate = 0.01
training_epochs = 20000
display_step = 500
batch_size = 41  # 41


def NeuralNetwork(dimensions=[64, 50, 25], no_class=4):
    """Build a deep Neural Network w/ tied weights.

        Parameters
        ----------
        dimensions : list, optional
            The number of neurons for each layer of the autoencoder.
        number of classes: integer, optional
        Returns
        -------
        X : Tensor
            Input placeholder to the network
        Y : Tensor
            Output softmax probabilities
        cost : Tensor
            Overall cost to use for training
        """
    X = tf.placeholder(tf.float32, shape=[None, dimensions[0]], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, no_class], name="Y")
    current_input = X

    # Build The classifier
    weights = []
    biases = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        weights.append(W)
        biases.append(b)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    # Create output Layer
    n_input = int(current_input.get_shape()[1])
    W = tf.Variable(
        tf.random_uniform([n_input, no_class],
                          -1.0 / math.sqrt(n_input),
                          1.0 / math.sqrt(n_input)))
    b = tf.Variable(tf.zeros([no_class]))
    weights.append(W)
    biases.append(b)
    output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)

    cost = tf.reduce_sum(tf.square(output - Y))
    cross_entropy = -tf.reduce_sum(Y * tf.log(output))
    return {'X': X, 'Y': Y, 'cost': cost, 'cross_entropy': cross_entropy,
            'model': output, 'W': weights, 'b': biases, 'predict': output}


net = NeuralNetwork(dimensions=[64, 50, 20])
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(net['cost'])
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Running Neural Network
for ep_i in range(training_epochs):
    avg_cost = 0
    total_batch = int(X_train.shape[0] / batch_size)
    for i in range(total_batch):
        b_x, b_y = NextBatch(batch_size)
        _, c = sess.run([optimizer, net['cost']], feed_dict={net['X']: b_x, net['Y']: b_y})
        avg_cost += c
    avg_cost /= total_batch
    NextBatch.batchIndex = 0
    if (ep_i % display_step == 0):
        print("Epoch %d : %f" % (ep_i, avg_cost))

print("Optimization Finished!")

"""End Of Model Creation"""

"""Model Validation"""

correct_prediction = tf.equal(tf.argmax(net['predict'], 1), tf.argmax(y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:",
      sess.run(accuracy,
               feed_dict={net['X']: X_test, net['Y']: y_test}))  # print(metrics.accuracy_score(y_test.ravel(), y))
# cm = metrics.confusion_matrix(y_test.ravel(), y)
# print(cm)

"""End of Model Validation"""

sess.close()

data = sio.loadmat('/media/hardik/DataPart/VisualArea_VideoResponseMapping.mat')
features = data.get('FEATURES')
mappingLabel = data.get('mappingLabel')
print(mappingLabel.shape)
