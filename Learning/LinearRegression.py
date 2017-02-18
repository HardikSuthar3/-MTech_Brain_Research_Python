import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.ion()
n_observations = 100
fig, ax = plt.subplots(1, 1)
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
ax.scatter(xs, ys)
fig.show()
plt.draw()

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
Y_pred = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_sum(tf.pow(Y - Y_pred, 2)) / (n_observations - 1)

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

n_epochs = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0
    for epoch_i in range(n_epochs):
        for x, y in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})
        print(training_cost)
        if (epoch_i % 20 == 0):
            ax.plot(xs, sess.run(Y_pred, feed_dict={X: xs}), 'k', alpha=epoch_i / n_epochs)
            fig.show()
            plt.draw()
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost
fig.show()
plt.waitforbuttonpress()
print(type(cost))