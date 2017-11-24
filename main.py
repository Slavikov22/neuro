import os.path

from mnist import MNIST


mndata = MNIST(os.path.abspath('data/'))
mndata.load_training()
mndata.load_testing()
