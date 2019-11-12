import numpy as np


class Flatten:
    def __init__(self, transpose = True):
        self.shape = {}
        self.params = {
            'transpose': transpose
        }
        self.contains_weights = False

    def has_weights(self):
        return self.contains_weights

    def forward_pass(self, Z, save_cache = False):
        shape = Z.shape
        if save_cache:
            self.shape = shape
        data = np.ravel(Z).reshape(shape[0], -1)
        if self.params['transpose']:
            data = data.T
        return data

    def backward_pass(self, Z):
        if self.params['transpose']:
            Z = Z.T
        return Z.reshape(self.shape)