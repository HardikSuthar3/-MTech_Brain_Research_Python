"""
For Training Neural Network one batch is one image features
"""

import scipy.io as sio
import numpy as np
import numpy.matlib as npmat
from sklearn.cross_validation import train_test_split, KFold
from sklearn import preprocessing
import tensorflow as tf
import math
from sklearn.svm import SVC, LinearSVC, NuSVC

# data = sio.loadmat("/home/hardik/Desktop/MTech_Project/MTechData/MingData/Feature/ClassificationFeature.mat")
# X = data['X']
# Y = data['Y']


# X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
#
# y_train = np.argmax(y_train, 1)
# y_test = np.argmax(y_test, 1)
#
# model1 = NuSVC(nu=0.5)
# model1.fit(X_train, y_train)
# print(model1.score(X_test, y_test))

"""Neural Network Configuration"""
# Parameters
learning_rate = 0.01
training_epochs = 150
display_step = 100

# Network Parameters
n_hidden_1 = 50  # 1st layer number of features
n_hidden_2 = 25  # 2nd layer number of features
n_hidden_3 = 10  # 2nd layer number of features
n_input = 64  # SURF data input (img shape: 28*28)
n_classes = 5  # SURF total classes (0-9 digits)

"""Create NN Model"""


def DNNClassifier(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def NeuralNetwork(dimensions=[64, 50, 25, 10], no_class=5):
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


net = NeuralNetwork(dimensions=[64, 100, 100])

optmizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(net['cost'])

sess = tf.Session()
sess.run(tf.initialize_all_variables())

"""Create DataSet for Classification"""

data = sio.loadmat('/media/hardik/DataPart/Hello.mat')

kf = KFold(400, 5, shuffle=True)

oneHotEncoder = preprocessing.LabelBinarizer()

oneHotEncoder.fit(np.linspace(1, 5, 5))

for train_index, test_index in kf:

    """Running Optimizer"""
    for epoch_i in range(training_epochs):
        avg_cost = 0
        """Creating Train Data For Classification"""

        for i in train_index:
            train_batch = data["features"][0, i]  # Feature Matrix
            if (train_batch.shape[0] == 0):
                continue
            lbVal = data["labels"][0, i]  # Natural Movie Number
            train_label = np.matlib.repmat(np.asarray(lbVal), 1, train_batch.shape[0]).transpose()  # Label Vector
            train_label = oneHotEncoder.transform(train_label)
            _, c = sess.run([optmizer, net['cost']], feed_dict={net['X']: train_batch, net['Y']: train_label})
            avg_cost += c
        avg_cost /= 400

    """Creating Test Data"""

    test_data = np.empty([0, 64])
    test_label = np.empty([1, 0])

    for i in test_index:
        tmp_x = data["features"][0, i]  # Feature Matrix
        lbVal = data["labels"][0, i]  # Natural Movie Number
        tmp_y = np.matlib.repmat(np.asarray(lbVal), 1, tmp_x.shape[0])  # Label Vector
        test_data = np.vstack((test_data, tmp_x))
        test_label = np.hstack((test_label, tmp_y))
    test_label = oneHotEncoder.transform(test_label.transpose())

    """Run Model On Test Data"""
    predicted_label = net['predict']
    correct_prediction = tf.equal(tf.argmax(predicted_label, 1), tf.argmax(test_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={net['X']: test_data, net['Y']: test_label}))



    # for i in range(data["features"].shape[1]):
    #     X = data["features"][0, i]
    #     val = data["labels"][0, i]
    #     Z = npmat.repmat(np.asarray(val), 1, X.shape[0])
    #     Y = np.concatenate((Y, Z), axis=1)
