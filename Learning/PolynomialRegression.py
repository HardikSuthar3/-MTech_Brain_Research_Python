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

Y_pred = tf.Variable(tf.random_normal([1]), name='bias')

for pow_i in range(1, 5):
    W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
    Y_pred = tf.add(Y_pred, tf.multiply(tf.pow(X, pow_i), W))

cost = tf.reduce_sum(tf.pow(Y - Y_pred, 2)) / (n_observations - 1)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    n_epochs = 1000
    sess.run(tf.global_variables_initializer())

    prev_training_cost = 0
    for ep_i in range(n_epochs):
        for x, y in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})
        print(training_cost)
        if (ep_i % 100 == 0):
            ax.plot(xs, Y_pred.eval(session=sess, feed_dict={X: xs, Y: ys}), 'k', alpha=ep_i / n_epochs)
            # fig.show()
            plt.draw()
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost

ax.set_ylim([-3, 3])
fig.show()
plt.waitforbuttonpress()
