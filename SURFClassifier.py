import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib as skflow
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_validation import *

# mnist = input_data.read_data_sets(
#     "/home/hardik/Desktop/MTech_Project/Scripts/Python"
#     "/BrainResearch/TF_PKMittal/python/MNIST_data/", one_hot=True)
#



data = sio.loadmat("/home/hardik/Desktop/MTech_Project/MTechData/MingData/SURFFeatures.mat")
XX = data['X']
yy = data['y']
y2 = data['y2']

X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.25)

# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 7749 #39
display_step = 1

# Network Parameters
n_hidden_1 = 100  # 1st layer number of features
n_hidden_2 = 100  # 2nd layer number of features
n_input = 64  # SURF data input (img shape: 28*28)
n_classes = 5  # SURF total classes (0-9 digits)

X = tf.placeholder(tf.float32, shape=[None, n_input])
y = tf.placeholder(tf.float32, shape=[None, n_classes])


def NextBatch(batchSize=39):
    data = X_train[NextBatch.batchIndex:NextBatch.batchIndex + batchSize, :], \
           y_train[
           NextBatch.batchIndex:NextBatch.batchIndex + batchSize,
           :]
    NextBatch.batchIndex += batchSize
    return data


NextBatch.batchIndex = 0


def loadData():
    data = sio.loadmat("/home/hardik/Desktop/MTech_Project/MTechData/MingData/SURFFeatures.mat")
    X = data['X']
    y = data['y']
    y2 = data['y2']
    return {'X': X, 'y': y, 'y2': y2}


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


# Define Weights Variable
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = DNNClassifier(X, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# cost = tf.reduce_mean(tf.pow(tf.sub(pred, y), 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init_op = tf.initialize_all_variables()
a = []

with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(X_train.shape[0] / batch_size)
        for i in range(total_batch):
            b_x, b_y = NextBatch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: b_x, y: b_y})
        NextBatch.batchIndex = 0
        print(c)
        avg_cost += c / total_batch
        # if epoch % display_step == 0:
        #     print("Epoch:", '%04d' % (epoch + 1), "cost=",
        #           "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: X_test, y: y_test}))
