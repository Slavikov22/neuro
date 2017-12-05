import os.path
import random

import numpy as np
from mnist import MNIST

from neuro.helpers.activate_functions import Logistic, Relu, Softmax, Softplus
from neuro.helpers.error_functions import MSE, CrossEntropy
from neuro.layers.hidden_layer import HiddenLayer
from neuro.layers.input_layer import InputLayer
from neuro.layers.output_layer import OutputLayer
from neuro.neural import Neural


def main():
    mndata = MNIST(os.path.abspath('data/'))
    mndata.load_training()
    mndata.load_testing()

    train_labels = list(mndata.train_labels)
    test_labels = list(mndata.test_labels)

    train_set = []
    for num in range(len(mndata.train_images)):
        x = np.array(mndata.train_images[num]) / 256.
        y = np.zeros((10, ))
        y[train_labels[num]] = 1
        train_set.append((x, y))

    test_set = []
    for num in range(len(mndata.test_images)):
        x = np.array(mndata.test_images[num]) / 256.
        y = np.zeros((10, ))
        y[test_labels[num]] = 1
        test_set.append((x, y))

    neural = Neural(offset_neuron=False)
    neural.add_layer(InputLayer(num_neurons=784))
    neural.add_layer(HiddenLayer(num_neurons=800, activate_func=Logistic))
    neural.add_layer(OutputLayer(num_neurons=10, activate_func=Logistic))
    neural.error_func = MSE

    random.shuffle(train_set)
    random.shuffle(test_set)

    neural.train(train_set=train_set[:1000], test_set=test_set[:1000], batch_size=30, epoch_count=40)


if __name__ == '__main__':
    main()
