import numpy as np
import math
import sys


class Relu:
    @staticmethod
    def compute(ar):
        return np.vectorize(lambda x: max(0, x))(ar)

    @staticmethod
    def diff(ar):
        return np.ones((len(ar), ))


''' Сглаженная функция Relu '''
class Softplus:
    @staticmethod
    def compute(ar):
        return np.vectorize(lambda x: math.log(1 + math.exp(x), math.exp(1)))(ar)

    @staticmethod
    def diff(ar):
        return np.vectorize(lambda x: 1.0 / (1 + math.exp(-x)))(ar)


class Softmax:
    @staticmethod
    def compute(ar):
        s = sum([math.exp(x) for x in ar])
        return [math.exp(x) / s for x in ar]

    @staticmethod
    def diff(ar):
        s = sum([math.exp(x) for x in ar])
        return [(math.exp(x) / s) * (1 - math.exp(x) / s) for x in ar]


class Logistic:
    @staticmethod
    def compute(ar):
        return np.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(ar)

    @staticmethod
    def diff(ar):
        return [x * (1 - x) for x in ar]
