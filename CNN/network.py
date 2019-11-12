from CNN.layers.activations import Relu, Elu, Softmax, Sigmoid
from CNN.layers.convolution import Convolution
from CNN.layers.fnlayer import FullyConnected
from CNN.layers.flattened import Flatten
from CNN.layers.maxpooling import Pooling

from CNN.utilities.model import Model
from CNN.utilities.loss import CategoricalCrossEntropy

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

train_data = train_data[:,:,:, np.newaxis]/255
test_data = test_data[:, :, :, np.newaxis]/255
# train_labels = train_labels[:, np.newaxis]
# test_labels = test_labels[:, np.newaxis]
train_labels = tf.one_hot(train_labels, depth=10)
train_labels = np.array(train_labels)
test_labels = tf.one_hot(test_labels, depth=10)
test_labels = np.array(test_labels)
# print(train_data[0])
shuffle_indices = np.random.randint(0, len(train_data), 10)
train_data = train_data[shuffle_indices, :,:,:]
train_labels = train_labels[shuffle_indices, :]
test_data = test_data[:1000, :,:,:]
test_labels = test_labels[:1000, :]
print("Train data shape: {}, {}".format(train_data.shape, train_labels.shape))
print("Test data shape: {}, {}".format(test_data.shape, test_labels.shape))
model = Model(
        Convolution(filters=5, padding='same'),
        Elu(),
        Pooling(mode='max', kernel_shape=(2, 2), stride=2),
        Flatten(),
        FullyConnected(units=10),
        Softmax(),
        name='cnn5'
    )
model.set_loss(CategoricalCrossEntropy)
# shuffled_indices = np.random.permutation(train_data.shape[0])
model.set_num_classes(10)
model.train(train_data, train_labels.T, epochs=2, batch_size=100)
# model.load_weights()
pred = model.predict(test_data)
print(pred.shape)
# predicts = np.argmax(model.predict(test_data), axis=0)
# test_labels = np.argmax(test_labels, axis=1)
# print(predicts)
# print(test_labels)
# a = [1 for x,y in zip(test_labels, predicts) if x==y]
# print(sum(a)/len(test_labels))
# print('Testing accuracy = {}'.format(model.evaluate(test_data, test_labels)))

