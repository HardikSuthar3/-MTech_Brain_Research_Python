"""
This script is to classify the response of mouse to six different type of movies
"""

import scipy.io as sio
import numpy as np
from sklearn.cross_validation import train_test_split
import tensorflow as tf

data = sio.loadmat("/home/hardik/Desktop/MTech_Project/MTechData/MingData/Feature/ClassificationFeature.mat")
X = data['X']
Y = data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

learning_rate = 0.001
training_epochs = 10
batch_size = 10
display_step = 1

n_hidden_1 = 27  # 1st layer number of features
n_hidden_2 = 27  # 2nd layer number of features
n_input = 9  # MNIST data input (img shape: 28*28)
n_classes = 6  # MNIST total classes (0-9 digits)

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


def multilayer_perceptron(x, weights, biases):
    x_float = tf.to_double(x)
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.nn.softmax((tf.matmul(layer_2, weights['h3']) + biases['b3']))
    return out_layer


# Construct model
y_pred = multilayer_perceptron(x, weights, biases)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))

cost = tf.reduce_mean(tf.pow(y_pred - y, 2))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init_op = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(X_train.shape[0] / batch_size)
        batch_index = 0
        for i in range(total_batch):
            # Partition Data Into batch
            batch_x = X_train[batch_index:batch_index + batch_size - 1, :]
            batch_y = y_train[batch_index:batch_index + batch_size - 1, :]
            batch_index = batch_index + batch_size

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            avg_cost += c
        if (epoch % display_step == 0):
            avg_cost = avg_cost / total_batch
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Done")

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))
