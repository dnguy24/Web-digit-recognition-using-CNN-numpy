import numpy as np
from CNN.utilities.management import get_layer_num, inc_layer_num

class Pooling:
    def __init__(self, kernel_shape=(3,3), stride=1, mode="max", name=None):
        """

        :param kernel_shape:
        :param stride:
        :param mode:
        """
        self.params = {
            'kernel_shape': kernel_shape,
            'stride': stride,
            'mode': mode
        }
        self.type = 'pooling'
        self.cache = {}
        self.contains_weights = False
        self.name = name

    def has_weights(self):
        return self.contains_weights

    def forward_pass(self, X, save_cache=False):
        """

        :param X:
        :param save_cache:
        :return:
        """
        (M, prev_H, prev_W, prev_D) = X.shape
        filter_H, filter_W = self.params['kernel_shape']

        n_H = int(1 + (prev_H - filter_H) / self.params['stride'])
        n_W = int(1 + (prev_W - filter_W) / self.params['stride'])
        n_C = prev_D

        A = np.zeros((M, n_H, n_W, n_C))

        for i in range(M):
            for h in range(n_H):
                for w in range(n_W):
                    vert_start = h*self.params['stride']
                    vert_end = vert_start + filter_H
                    horiz_start = w * self.params['stride']
                    horiz_end = horiz_start + filter_W

                    for c in range(n_C):
                        if self.params['mode'] == 'average':
                            A[i, h, w, c] = np.mean(X[i, vert_start: vert_end, horiz_start:horiz_end, c])
                        else:
                            A[i, h, w, c] = np.max(X[i, vert_start: vert_end, horiz_start:horiz_end, c])
        if save_cache:
            self.cache['A'] = X

        return A

    def distribute_value(self, dz, shape):
        """
            Distributes the input value in the matrix of dimension shape

            Arguments:
            dz -- input scalar
            shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

            Returns:
            a -- Array of size (n_H, n_W) for which we distributed the value of dz
            """
        (n_H, n_W) = shape
        average = dz / (n_H * n_W)

        a = np.ones(shape) * average
        return a

    def create_mask_from_window(self, x):
        mask = x == np.max(x)

        return mask

    def backward_pass(self, dA):
        A = self.cache['A']
        filter_H, filter_W = self.params['kernel_shape']

        (M, prev_H, prev_W, prev_D) = A.shape
        (m, n_H, n_W, n_C) = dA.shape

        dA_prev = np.zeros(A.shape)

        for i in range(M):
            a = A[i]

            for h in range(n_H):
                for w in range(n_W):
                    vert_start = h * self.params['stride']
                    vert_end = vert_start + filter_H
                    horiz_start = w * self.params['stride']
                    horiz_end = horiz_start + filter_W

                    for c in range(n_C):
                        if self.params['mode'] == 'max':
                            a_slice = a[vert_start : vert_end, horiz_start : horiz_end, c]
                            mask = self.create_mask_from_window(a_slice)
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        elif self.params['mode'] == 'average':
                            da = dA[i, h, w, c]
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += self.distribute_value(da, self.params['kernel_shape'])


        return dA_prev