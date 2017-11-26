import numpy as np


class Layer:
    def __init__(self, num_neurons, activate_func):
        self.num_neurons = num_neurons
        self.activate_func = activate_func

        self.y = None
        self.s = None
