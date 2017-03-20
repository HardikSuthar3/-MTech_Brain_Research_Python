import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn.decomposition import pca
from sklearn.model_selection import train_test_split


def get_hog_features():
    sess = tf.Session()
    data = sio.loadmat(
        '/home/hardik/Desktop/MTech_Project/Data/FeatureVerificationData/HOG_Experiment/NeuralNetworkData.mat')
    print(data.keys())
    features = data['features']
    labels = data['labels'].flatten() - 1
    labels = sess.run(tf.one_hot(labels, depth=4))
    print(labels.shape)

    # Normalizing The Data
    from sklearn.preprocessing import Normalizer
    normalized_features = Normalizer().fit(features).transform(features)

    # Dimension Reduction
    pcaModel = pca.PCA(n_components=50)
    pca_features = pcaModel.fit_transform(normalized_features)
    sess.close()
    return {'features': pca_features, 'labels': labels}


def NextBatch():
    s = NextBatch.batchSize
    e = s + batch_size
    NextBatch.batchSize += batch_size
    return trainX[s:e, :], trainY[s:e, :]


# Build Neural Network
def NNModl(dimensions=[64, 50, 25, 10], n_class=4):
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


NextBatch.batchSize = 0

data = get_hog_features()

trainX, testX, trainY, testY = train_test_split(data['features'], data['labels'], train_size=0.8)

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
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for iter in range(50):
        NextBatch.batchSize = 0
        for ep_i in range(int(trainX.shape[0] / batch_size)):
            X, Y = NextBatch()
            sess.run(optimizer, feed_dict={
                NN['x']: X,
                NN['y']: Y
            })
            training_cost = sess.run(loss, feed_dict={
                NN['x']: trainX,
                NN['y']: trainY
            })
            if (ep_i % 50 == 0):
                print("Training Cost: %f" % training_cost)
            if (np.abs(prev_training_cost - training_cost) < 0.00001):
                print("Exiting")
                break
            prev_training_cost = training_cost

        Accuracy = sess.run(accuracy, feed_dict={
            NN['x']: testX,
            NN['y']: testY
        })
        modelSaver.save(sess=sess, save_path='/home/hardik/Desktop/MTech_Project/Scripts/Python'
                                             '/MTech_Brain_Research_Python/Hardik_Implementation'
                                             '/Verification_Experiment/HOG_Experiment/SavedModel/nm_hog.ckpt',
                        global_step=iter)

    print("Accuracy: %f" % Accuracy)
