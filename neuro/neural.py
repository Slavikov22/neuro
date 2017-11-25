import numpy as np


class Neural:
    def __init__(self):
        self.layers = []
        self.weights = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, train_set):
        self.init_weight()

        for (x, y) in train_set:
            self.correct(x, y)

    def correct(self, x, y):
        # Прямое распространение
        for num_layer in range(len(self.layers) - 1):
            x = self.weights[num_layer].dot(x)
            x = np.vectorize(self.layers[num_layer].activate_func)(x)

        e = np.sqrt(np.power(y - x, 2))  # Вектор ошибки

        # Обратное распространение ошибки
        for num_layer in reversed(range(len(self.layers) - 1)):
            e = self.weights[num_layer].transpose().dot(e)

    def init_weight(self):
        self.weights = []
        self.weights.append(np.array([0.9, 0.3, 0.4, 0.2, 0.8, 0.2, 0.1, 0.5, 0.6]).reshape((3, 3)))
        self.weights.append(np.array([0.3, 0.7, 0.5, 0.6, 0.5, 0.2, 0.8, 0.1, 0.9]).reshape((3, 3)))

        # for num_layer in range(len(self.layers) - 1):
        #     rows = self.layers[num_layer + 1].num_neurons
        #     columns = self.layers[num_layer].num_neurons
        #     self.weights.append(np.random.sample((rows, columns)))
