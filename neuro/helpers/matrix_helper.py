import numpy as np


def get_random_matrix(shape, center, max_offset):
    offset_matrix = np.random.sample(shape) * 2 - 1  # [-1, 1]
    offset_matrix *= max_offset  # [-max_offset, max_offset]

    center_matrix = np.ones(shape) * center

    return center_matrix + offset_matrix
