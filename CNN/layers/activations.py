import numpy as np
class Relu:
    def __init__(self):
        self.cache = {}
        self.contains_weights = False

    def has_weights(self):
        return self.contains_weights

    def forward_pass(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        X = np.where(Z>1e-8, Z, 0)
        return X

    def backward_pass(self, dA):
        Z = self.cache['Z']
        dA = dA / dA.shape[1]
        Z = Z / Z.shape[1]
        return dA * np.where(Z >1e-8, 1, 0)

class Elu:
    def __init__(self, alpha = 1.2):
        self.cache = {}
        self.params = {
            'alpha':alpha
        }
        self.contains_weights = False

    def has_weights(self):
        return self.contains_weights

    def forward_pass(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        X = np.where(Z>0, Z, self.params['alpha']*(np.exp(Z) - 1))
        return X

    def backward_pass(self, dA):
        Z = self.cache['Z']
        return dA * np.where(Z >0, 1, self.params['alpha'] * np.exp(Z))

class Softmax:
    def __init__(self):
        self.cache = {}
        self.contains_weights = False

    def has_weights(self):
        return self.contains_weights

    def forward_pass(self, Z, save_cache = False):
        if save_cache:
            self.cache['Z'] = Z
        b = Z.max()
        e = np.exp(Z - b)
        s = np.sum(e, axis=0, keepdims=True)
        return e / s

    def backward_pass(self, dA):
        Z = self.cache['Z']
        return dA * (Z * (1 - Z))

class Sigmoid:
    def __init__(self):
        self.cache = {}
        self.contains_weights = False

    def has_weights(self):
        return self.contains_weights

    def forward_pass(self, Z, save_cache = False):
        if save_cache:
            self.cache['Z'] = Z
        return 1/(1+np.exp(-Z))

    def backward_pass(self, dA):
        Z = self.cache['Z']
        s = self.forward_pass(Z)*(1-self.forward_pass(Z))
        return dA * s

