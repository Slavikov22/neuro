import numpy as np
import math


class Neural:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.step = 0.01
        self.eps = 0.000001

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, train_set):
        self.init_weight()

        e = 0
        for (x, y) in train_set:
            new_e = 1
            while math.fabs(new_e - e) > self.eps:
                new_e = self.correct(x, y)
                print('Error: ' + str(new_e) + '   Diff: ' + str(e - new_e))
            e = new_e

    def correct(self, x, y0):
        y = np.copy(x)
        self.layers[0].s = np.copy(y)
        self.layers[0].y = np.copy(y)

        # Прямое распространение
        for num_layer in range(1, len(self.layers)):
            y = self.weights[num_layer - 1].T.dot(y)  # Высчитываем Si
            self.layers[num_layer].s = np.copy(y)
            y = np.vectorize(self.layers[num_layer].activate_func)(y)  # Высчитываем Yi
            self.layers[num_layer].y = np.copy(y)

        # Метод градиентного спуска
        sigma = (self.layers[-1].y - y0) * self.layers[-1].y * (1 - self.layers[-1].y)
        for num_layer in reversed(range(0, len(self.layers) - 1)):
            delta_w = -self.step * np.outer(self.layers[num_layer].y, sigma)
            self.weights[num_layer] += delta_w
            sigma = np.sum((sigma * self.weights[num_layer]), axis=1) \
                    * self.layers[num_layer].y * (1 - self.layers[num_layer].y)  # dYi/dSj

        e = 0.5 * np.sqrt(np.power(y - y0, 2)).sum()
        return e

    def init_weight(self):
        self.weights = []

        for num_layer in range(len(self.layers) - 1):
            rows = self.layers[num_layer].num_neurons
            columns = self.layers[num_layer + 1].num_neurons
            self.weights.append(np.random.sample((rows, columns)))
