import os.path
import numpy as np
import sys

from mnist import MNIST
from neuro.neural import Neural
from neuro.layers.input_layer import InputLayer
from neuro.layers.output_layer import OutputLayer
from neuro.layers.hidden_layer import HiddenLayer
from neuro.activate_functions import logistic


def main():
    mndata = MNIST(os.path.abspath('data/'))
    mndata.load_training()
    mndata.load_testing()

    train_labels = list(mndata.train_labels)

    train_set = []
    for num in range(len(mndata.train_images)):
        x = np.array(mndata.train_images[num]) / 256.
        y = np.zeros((10, ))
        y[train_labels[num]] = 1
        train_set.append((x, y))

    neural = Neural(offset_neuron=False)
    neural.add_layer(InputLayer(num_neurons=784, activate_func=logistic))
    neural.add_layer(HiddenLayer(num_neurons=800, activate_func=logistic))
    neural.add_layer(OutputLayer(num_neurons=10, activate_func=logistic))

    neural.train(train_set=train_set[:1000], test_set_len=100, epoch_count=400)

    print(neural.weights)


if __name__ == '__main__':
    main()
