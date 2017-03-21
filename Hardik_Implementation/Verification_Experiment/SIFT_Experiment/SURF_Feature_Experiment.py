import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import batch_norm
import sklearn.preprocessing as preprocess


def LoadData():
    # Load and Split the Data
    data = sio.loadmat('/home/hardik/Desktop/MTech_Project/Data/FeatureVerificationData/NeuralNetworkData.mat')
    xs = data['features']
    ys = data['labels']

    # Convert label from [1,4] -> [0,3]
    ys = (ys - 1).ravel()

    sess = tf.Session()
    ys = sess.run(tf.one_hot(indices=ys, depth=4, on_value=1, off_value=0))

    trX, tsX, trY, tsY = train_test_split(xs, ys, train_size=0.8)
    sess.close()
    return trX, tsX, trY, tsY


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


def getAreaData(areaNumer):
    fileName = '/home/hardik/Desktop/MTech_Project/Data/FeatureVerificationData/area%d.mat' % areaNumer
    data = sio.loadmat(file_name=fileName)
    normalizer = preprocess.Normalizer()
    features = normalizer.fit_transform(data['features'])
    labels = data['labels'].ravel() - 1  # [1,4] to [0,3] range conversion

    sess = tf.Session()
    y2 = sess.run(tf.one_hot(indices=labels, depth=4, on_value=1, off_value=0))
    sess.close()

    return {'x': features, 'y': labels, 'y2': y2}


# Configure Neural Network
learning_rate = 0.01
batch_size = 60
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
    modelSaver.restore(sess=sess,
                       save_path='/home/hardik/Desktop/MTech_Project/Scripts/Python/MTech_Brain_Research_Python'
                                 '/Hardik_Implementation/Verification_Experiment/SavedModel/d1/nm_surf.ckpt-499')
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
               '/Hardik_Implementation/Verification_Experiment/surf_result.txt', visualAreaPosteriorValues, fmt='%f')
