import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import pca
from sklearn.model_selection import train_test_split
import scipy.io as sio

"""Transforms the Raw HOG features into Normalized PCA features for better classification"""

def load_data():
    data = sio.loadmat('/home/hardik/Desktop/MTech_Project/Data/HOG_Feature_Data/natural_movies_hog_features.mat')
    features = data['hog_features']
    labels = data['hog_labels'].flatten()
    from sklearn.preprocessing import Normalizer
    normalized_features = Normalizer().fit(features).transform(features)
    pcaModel = pca.PCA(n_components=50)
    pca_features = pcaModel.fit_transform(normalized_features)
    return {'features': pca_features, 'labels': labels}

data = load_data()
x_train, x_test, y_train, y_test = train_test_split(data['features'], data['labels'], train_size=0.8)

def NextBatch(batchSize=16):
    data = x_train[NextBatch.batchIndex:NextBatch.batchIndex + batchSize, :], \
           y_train[
           NextBatch.batchIndex:NextBatch.batchIndex + batchSize,
           :]
    NextBatch.batchIndex += batchSize
    return data

NextBatch.batchIndex = 0

"""Create a Neural Network Model"""
learning_rate = 0.01
training_epochs = 20000
display_step = 100
batch_size = 16

def NNModel(dimensions=[50, 25, 10], n_class=4):
    x = tf.placeholder(tf.float32, shape=[None, dimensions[0]])
    y = tf.placeholder(tf.float32, shape=[None, dimensions[0]])
    current_input = x

    # Build The classifier
    weights = []
    biases = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = current_input.get_shape()[1]
        w = tf.Variable(tf.random_normal([n_input, n_output]), tf.float32)
        b = tf.Variable(tf.zeros([n_output]), tf.float32)
        weights.append(w)
        biases.append(b)
        current_input = tf.nn.tanh(tf.add(tf.matmul(current_input, w), b))
    # Creating Output Layer
    n_input = int(current_input.get_shape()[1])
    w = tf.Variable(tf.random_normal(shape=[n_input, n_class]), dtype=tf.float32)
    b = tf.Variable(tf.zeros(shape=[n_class]), dtype=tf.float32)
    output = tf.add(tf.matmul(current_input, w), b)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, y))
    return {'x': x, 'y': y, 'weights': weights, 'biases': biases, 'cross_entropy': cross_entropy, 'output': output}
