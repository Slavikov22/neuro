import random
import numpy as np

from neuro.helpers.matrix_helper import get_random_matrix


class Neural:
    def __init__(self):
        self.layers = []
        self.step = 0.002
        self.moment = 0.001
        self.regress = 0.001
        self.error_func = None

        self.delta_w = []

    ''' Тренировать нейросеть'''
    def train(self, train_set, test_set, epoch_count=1, batch_size=1):
        self.init_weight()
        self.delta_w = [np.zeros(self.layers[num_layer].weights.shape) for num_layer in range(len(self.layers) - 1)]

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
            print(' '*4 + 'Percent of errors: {}%'.format(num_errors / len(test_set) * 100))
            print('-'*40)

    '''Корректировка весов'''
    def correct_weights(self, batch):
        batch_delta_w = [np.zeros(self.layers[num_layer].weights.shape) for num_layer in range(len(self.layers) - 1)]

        for (x, y) in batch:
            self.go_forward(x)
            cur_delta_w = self.get_delta_weights(y)
            for num_layer in range(len(self.layers) - 1):
                batch_delta_w[num_layer] += cur_delta_w[num_layer]

        for num_layer in range(len(self.layers) - 1):
            layer = self.layers[num_layer]
            layer.weights -= batch_delta_w[num_layer]

        self.delta_w = batch_delta_w.copy()

    '''Прямое распространение'''
    def go_forward(self, x):
        ar = np.copy(x)

        self.layers[0].y = np.copy(ar)

        for num_layer in range(1, len(self.layers)):
            prev_layer = self.layers[num_layer - 1]
            layer = self.layers[num_layer]

            ar = prev_layer.weights.T.dot(ar)  # Высчитываем Si
            ar = layer.activate_func.compute(ar)  # Высчитываем Yi

            layer.y = np.copy(ar)

    '''Метод градиентного спуска'''
    def get_delta_weights(self, y0):
        delta_weights = []

        dEdY = self.error_func.diff(self.layers[-1].y, y0)
        dYdS = self.layers[-1].activate_func.diff(self.layers[-1].y)
        sigma = dEdY * dYdS
        for num_layer in reversed(range(0, len(self.layers) - 1)):
            layer = self.layers[num_layer]
            delta_weights.append(
                self.step * (np.outer(layer.y, sigma) + self.regress * self.layers[num_layer].weights) +
                self.moment * self.delta_w[num_layer]
            )
            dYdS = layer.activate_func.diff(layer.y)
            sigma = np.sum((sigma * layer.weights), axis=1) * dYdS

        return list(reversed(delta_weights))

    ''' Вернуть ответ нейросети '''
    def get_result(self, x):
        self.go_forward(x)
        return self.layers[-1].y

    ''' Вернуть ошибку '''
    def get_error(self, x, y0):
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
                center=0,
                max_offset=0.5
            )

    def add_layer(self, layer):
        self.layers.append(layer)
