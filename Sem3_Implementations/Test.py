import tensorflow as tf
import numpy as np
import tensorflow.contrib.losses as losses
import tensorflow.contrib.metrics as metrics
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

cross_entropy = losses.mean_squared_error(y, y_)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()

saver = tf.train.Saver()

sess.run(init)

for i in range(550):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

fileName = "/home/hardik/Desktop/MTech_Project/Scripts/Python/Brain_Research_Python/MNIST_data/myModel.ckpt"
savePath = saver.save(sess, fileName)


# cm = metrics.confusion_matrix(tf.argmax(y, 1), tf.argmax(y_, 1))
# C = sess.run(cm, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
# print(C)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
