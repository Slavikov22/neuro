import random

import numpy as np

from neuro.helpers.matrix_helper import get_random_matrix


class Neural:
    def __init__(self, offset_neuron=False):
        self.offset_neuron = offset_neuron
        self.layers = []
        self.step = 0.00001
        self.error_func = None

    ''' Тренировать нейросеть'''
    def train(self, train_set, test_set, epoch_count=1, batch_size=1):
        self.init_weight()

        for epoch_num in range(epoch_count):
            cur_train_set = train_set[:]
            random.shuffle(cur_train_set)

            while len(cur_train_set) > 0:
                cur_batch_size = min(batch_size, len(cur_train_set))
                self.correct_weights(cur_train_set[:cur_batch_size])
                cur_train_set = cur_train_set[cur_batch_size:]

            average_error = self.get_average_error(test_set)
            num_errors = self.get_num_errors(test_set)

            print('Epoch {}:'.format(epoch_num))
            print(' '*4 + 'Average error: {}'.format(average_error))
            print(' '*4 + 'Number of errors: {}/{}'.format(num_errors, len(test_set)))
            print('-'*40)

    '''Корректировка весов'''
    def correct_weights(self, batch):
        delta_w = [np.zeros(self.layers[num_layer].weights.shape) for num_layer in range(len(self.layers) - 1)]

        for (x, y) in batch:
            self.go_forward(x)
            cur_delta_w = self.get_delta_weights(y)
            for num_layer in range(len(self.layers) - 1):
                delta_w[num_layer] += cur_delta_w[num_layer]

        for num_layer in range(len(self.layers) - 1):
            layer = self.layers[num_layer]
            layer.weights -= delta_w[num_layer]

    '''Прямое распространение'''
    def go_forward(self, x):
        ar = np.copy(x)

        if self.offset_neuron:
            ar = np.concatenate(([1], ar))

        ar = self.layers[0].activate_func.compute(ar)
        self.layers[0].y = np.copy(ar)

        for num_layer in range(1, len(self.layers)):
            prev_layer = self.layers[num_layer - 1]
            layer = self.layers[num_layer]

            ar = prev_layer.weights.T.dot(ar)  # Высчитываем Si
            ar = layer.activate_func.compute(ar)  # Высчитываем Yi

            if self.offset_neuron:
                ar[0] = 1

            layer.y = np.copy(ar)

    '''Метод градиентного спуска'''
    def get_delta_weights(self, y0):
        result = []

        if self.offset_neuron:
            y0 = np.concatenate(([1], y0))

        dEdY = self.error_func.diff(self.layers[-1].y, y0)
        dYdS = self.layers[-1].activate_func.diff(self.layers[-1].y)
        sigma = dEdY * dYdS
        for num_layer in reversed(range(0, len(self.layers) - 1)):
            layer = self.layers[num_layer]
            result.append(self.step * np.outer(layer.y, sigma))
            dYdS = layer.activate_func.diff(layer.y)
            sigma = np.sum((sigma * layer.weights), axis=1) * dYdS

        return list(reversed(result))

    ''' Вернуть ответ нейросети '''
    def get_result(self, x):
        self.go_forward(x)
        return self.layers[-1].y

    ''' Вернуть ошибку '''
    def get_error(self, x, y0):
        if self.offset_neuron:
            y0 = np.concatenate(([1], y0))

        return self.error_func.get_error(self.get_result(x), y0)

    ''' Вернуть количество ошибок в наборе тестов'''
    def get_num_errors(self, test_set):
        num_errors = 0
        for (x, y) in test_set:
            result = self.get_result(x)

            correct_digit = np.argmax(y)
            result_digit = np.argmax(result)

            if correct_digit != result_digit:
                num_errors += 1

        return num_errors

    ''' Вернуть среднюю ошибку по набору тестов'''
    def get_average_error(self, test_set):
        sum_error = 0
        for (x, y) in test_set:
            error = self.get_error(x, y)
            sum_error += error

        return sum_error / len(test_set)

    def init_weight(self):
        for num_layer in range(len(self.layers) - 1):
            layer = self.layers[num_layer]
            next_layer = self.layers[num_layer + 1]
            layer.weights = get_random_matrix(
                shape=(layer.num_neurons, next_layer.num_neurons),
                center=0.5,
                max_offset=0.2
            )

            if self.offset_neuron:
                layer.weights[:, 0] = np.zeros((layer.num_neurons,))

    def add_layer(self, layer):
        if self.offset_neuron:
            layer.num_neurons += 1

        self.layers.append(layer)
