from neuro.helpers.activate_functions import Relu


class Layer:
    def __init__(self, num_neurons, activate_func=Relu):
        self.num_neurons = num_neurons
        self.activate_func = activate_func
        self.weights = []

        self.y = None
