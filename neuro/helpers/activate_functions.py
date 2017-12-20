import numpy as np
import math


class Logistic:
    @staticmethod
    def compute(ar):
        return np.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(ar)

    @staticmethod
    def diff(ar):
        return Logistic.compute(ar) * (1 - Logistic.compute(ar))
