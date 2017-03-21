import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn.decomposition import pca
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


# Build Neural Network
def NNModl(dimensions=[50, 100, 25, 10], n_class=4):
    X = tf.placeholder(dtype=tf.float32, shape=[None, dimensions[0]])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_class])
    current_input = X

    # Making The Classifier
    weights = []
    biases = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        w = tf.Variable(tf.random_normal(shape=[n_input, n_output]), dtype=tf.float32, name='Weights_%d' % layer_i)
        b = tf.Variable(tf.zeros([n_output]), dtype=tf.float32, name='Biases_%d' % layer_i)
        weights.append(w)
        biases.append(b)
        current_input = tf.nn.tanh(tf.add(tf.matmul(current_input, w), b))

    # Create Output Classification Layer
    n_input = int(current_input.get_shape()[1])
    n_output = n_class
    w = tf.Variable(tf.random_normal([n_input, n_output]), name='Weights_Output', dtype=tf.float32)
    b = tf.Variable(tf.zeros([n_output]), name='Biases_Output', dtype=tf.float32)

    weights.append(w)
    biases.append(b)

    output = tf.add(tf.matmul(current_input, w), b)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))

    return {'x': X, 'y': Y, 'weights': weights, 'biases': biases, 'cross_entropy': cross_entropy, 'output': output}


def getAreaData(areaNumer):
    fileName = '/home/hardik/Desktop/MTech_Project/Data/FeatureVerificationData/HOG_Experiment/area%d.mat' % areaNumer
    sess = tf.Session()
    data = sio.loadmat(fileName)
    # print(data.keys())
    features = data['features']
    labels = data['labels'].flatten() - 1
    y2 = sess.run(tf.one_hot(labels, depth=4))

    # Normalizing The Data
    normalized_features = Normalizer().fit(features).transform(features)

    # Dimension Reduction
    pca_features = pcaModel.transform(normalized_features)
    sess.close()
    return {'x': pca_features, 'y': labels, 'y2': y2}


sess = tf.Session()
data = sio.loadmat(
    '/home/hardik/Desktop/MTech_Project/Data/FeatureVerificationData/HOG_Experiment/NeuralNetworkData.mat')
features = data['features']
labels = data['labels'].flatten() - 1
labels = sess.run(tf.one_hot(labels, depth=4))

# Normalizing The Data
normalized_features = Normalizer().fit(features).transform(features)

# Dimension Reduction
pcaModel = pca.PCA(n_components=50)
pca_features = pcaModel.fit_transform(normalized_features)
sess.close()

# Configure Neural Network
learning_rate = 0.01
batch_size = 32
training_epochs = 150
display_step = 100

# Create Neural Network Object
NN = NNModl()
output = NN['output']
loss = NN['cross_entropy']
y_pred = tf.nn.softmax(output)

correct_prediction = tf.equal(tf.arg_max(input=NN['y'], dimension=1), tf.arg_max(input=y_pred, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimization Process
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

# Create Model Saver Object to Save Model
modelSaver = tf.train.Saver()

with tf.Session() as sess:
    modelSaver.restore(sess,
                       save_path='/home/hardik/Desktop/MTech_Project/Scripts/Python/MTech_Brain_Research_Python'
                                 '/Hardik_Implementation/Verification_Experiment/HOG_Experiment/SavedModel/nm_hog.ckpt-9')

    visualAreaPosteriorValues = []
    for areaNumber in range(1, 10):
        data = getAreaData(areaNumber)

        posterior = 0

        result = sess.run(y_pred, feed_dict={
            NN['x']: data['x'],
            NN['y']: data['y2']
        })

        for row, j in enumerate(data['y']):
            posterior += result[row, j]

        posterior /= result.shape[0]

        visualAreaPosteriorValues.append(posterior)

        Accuracy = sess.run(accuracy, feed_dict={
            NN['x']: data['x'],
            NN['y']: data['y2']
        })
        print(Accuracy)

    visualAreaPosteriorValues = np.array(visualAreaPosteriorValues)

    np.savetxt('/home/hardik/Desktop/MTech_Project/Scripts/Python/MTech_Brain_Research_Python'
               '/Hardik_Implementation/Verification_Experiment/hog_result.txt', visualAreaPosteriorValues, fmt='%f')
