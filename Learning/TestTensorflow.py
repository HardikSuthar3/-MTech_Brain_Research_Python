import tensorflow as tf
import numpy as np

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
print(sess.run(x))
