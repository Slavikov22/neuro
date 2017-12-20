import numpy as np


class MSE:
    @staticmethod
    def get_error(y, y0):
        return 0.5 * np.sqrt(np.power(y - y0, 2)).sum()

    @staticmethod
    def diff(y, y0):
        return y - y0
