import numpy as np
import math
import random


class Neural:
    def __init__(self, offset_neuron=False):
        self.offset_neuron = offset_neuron

        self.layers = []
        self.weights = []
        self.delta_w = []
        self.train_set = []
        self.test_set = []
        self.step = 0.01
        self.moment = 0.05

    def train(self, train_set, test_set_len=0, epoch_count=1):
        self.init_weight()

        self.test_set = train_set[:test_set_len]
        self.train_set = train_set[test_set_len:]

        for epoch_num in range(epoch_count):
            random.shuffle(self.train_set)
            for (x, y) in self.train_set:
                self.correct(x, y)

            print('Epoch {0}: {1}'.format(epoch_num, self.get_test_set_error()))

    '''Корректировка весов'''
    def correct(self, x, y0):
        self.go_forward(x)
        self.go_grad_descend(y0)

    '''Прямое распространение'''
    def go_forward(self, x):
        y = np.copy(x)

        if self.offset_neuron:
            y = np.concatenate(([1], y))

        self.layers[0].y = np.copy(y)

        for num_layer in range(1, len(self.layers)):
            y = self.weights[num_layer - 1].T.dot(y)  # Высчитываем Si
            y = np.vectorize(self.layers[num_layer].activate_func)(y)  # Высчитываем Yi

            if self.offset_neuron:
                y[0] = 1

            self.layers[num_layer].y = np.copy(y)

    '''Метод градиентного спуска'''
    def go_grad_descend(self, y0):
        if self.offset_neuron:
            y0 = np.concatenate(([1], y0))

        sigma = (self.layers[-1].y - y0) * self.layers[-1].y * (1 - self.layers[-1].y)
        for num_layer in reversed(range(0, len(self.layers) - 1)):
            delta_w = self.step * np.outer(self.layers[num_layer].y, sigma) + self.moment * self.moment * self.delta_w[num_layer]
            self.weights[num_layer] -= delta_w
            self.delta_w[num_layer] = delta_w
            sigma = np.sum((sigma * self.weights[num_layer]), axis=1) \
                    * self.layers[num_layer].y * (1 - self.layers[num_layer].y)  # dYi/dSj

    def get_error(self, x, y0):
        if self.offset_neuron:
            y0 = np.concatenate(([1], y0))

        self.go_forward(x)
        return 0.5 * np.sqrt(np.power(self.layers[-1].y - y0, 2)).sum()

    def get_test_set_error(self):
        sum_error = 0
        for (x, y) in self.test_set:
            error = self.get_error(x, y)
            sum_error += error

        return sum_error / len(self.test_set)

    def init_weight(self):
        self.weights = []
        self.delta_w = []

        for num_layer in range(len(self.layers) - 1):
            rows = self.layers[num_layer].num_neurons
            columns = self.layers[num_layer + 1].num_neurons

            self.weights.append(np.random.sample((rows, columns)))
            self.delta_w.append(np.zeros((rows, columns)))

            if self.offset_neuron:
                self.weights[-1][:, 0] = np.zeros((rows, ))

    def add_layer(self, layer):
        if self.offset_neuron:
            layer.num_neurons += 1

        self.layers.append(layer)
