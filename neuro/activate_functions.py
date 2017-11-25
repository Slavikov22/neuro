import math


def relu(x):
    return max(0, x)


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))
