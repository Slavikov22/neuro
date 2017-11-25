import os.path
import numpy as np

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

    neural = Neural()
    neural.add_layer(InputLayer(num_neurons=3, activate_func=logistic))
    neural.add_layer(HiddenLayer(num_neurons=3, activate_func=logistic))
    neural.add_layer(OutputLayer(num_neurons=3, activate_func=logistic))

    train_set = [(np.array([0.9, 0.1, 0.8]), np.array([0.5, 0.1, 0.2]))]
    neural.train(train_set=train_set)


if __name__ == '__main__':
    main()
