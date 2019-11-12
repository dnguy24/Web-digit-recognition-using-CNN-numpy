import numpy as np
from CNN.utilities.utils import random_initializer, zero_pad, glorot_uniform, random_ini
from CNN.utilities.management import get_layer_num, inc_layer_num
import pickle
from os import makedirs, remove, path

class Convolution:
    def __init__(self, filters, kernel_shape = (3,3), padding = 'valid', stride = 1, name = None):
        self.params = {
            'filters': filters,
            'padding': padding,
            'kernel_shape': kernel_shape,
            'stride': stride
        }
        self.cache = {}
        self.rmsprop_cache = {}
        self.momentum_cache = {}
        self.grads = {}
        self.contains_weights = True
        self.name = name
        self.type = 'conv'


    def has_weights(self):
        return self.contains_weights

    def save_weights(self, dump_path):
        dump_cache = {
            'cache': self.cache,
            'grads': self.grads,
            'momentum': self.momentum_cache,
            'rmsprop': self.rmsprop_cache
        }
        save_path = path.join(dump_path, self.name+'.pickle')
        makedirs(path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as d:
            pickle.dump(dump_cache, d)

    def load_weights(self, dump_path):
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            inc_layer_num(self.type)
        read_path = path.join(dump_path, self.name+'.pickle')
        with open(read_path, 'rb') as r:
            dump_cache = pickle.load(r)
        self.cache = dump_cache['cache']
        self.grads = dump_cache['grads']
        self.momentum_cache = dump_cache['momentum']
        self.rmsprop_cache = dump_cache['rmsprop']

    def conv_single_step(self, input, W, b):
        s = np.multiply(input, W) + b
        return np.sum(s)


    def forward_pass(self, X, save_cache = False):
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
        (M, prev_H, prev_W, prev_D) = X.shape
        filter_H, filter_W = self.params['kernel_shape']
        if 'W' not in self.params:
            print("not")
            shape = (filter_H, filter_W, prev_D, self.params['filters'])
            self.params['W'], self.params['b'] = random_ini(shape=shape)
        if self.params['padding'] == 'same':
            pad_h = int(((prev_H - 1)*self.params['stride'] + filter_H - prev_H) / 2)
            pad_w = int(((prev_W - 1)*self.params['stride'] + filter_W - prev_W) / 2)
            n_H = prev_H
            n_W = prev_W
        else:
            pad_h = 0
            pad_w = 0
            n_H = int((prev_H - filter_H) / self.params['stride']) + 1
            n_W = int((prev_W - filter_W) / self.params['stride']) + 1

        self.params['pad_h'], self.params['pad_w'] = pad_h, pad_w

        Z = np.zeros((M, n_H, n_W, self.params['filters']))

        X_pad = zero_pad(X, (pad_h, pad_w))

        for i in range(M):
            x = X_pad[i]
            for h in range(n_H):
                for w in range(n_W):
                        vert_start = h*self.params['stride']
                        vert_end = vert_start + filter_H
                        horiz_start = self.params['stride'] * w
                        horiz_end = horiz_start + filter_W
                        for c in range(self.params['filters']):
                            x_slice = x[vert_start: vert_end, horiz_start: horiz_end, :]
                            Z[i, h, w, c] = self.conv_single_step(x_slice, self.params['W'][:, :, :, c],
                                                              self.params['b'][:, :, :, c])
        if save_cache:
            self.cache['A'] = X

        return Z

    def init_cache(self):
        cache = dict()
        cache['dW'] = np.zeros_like(self.params['W'])
        cache['db'] = np.zeros_like(self.params['b'])
        return cache

    def backward_pass(self, dZ):
        A = self.cache['A']
        (M, prev_H, prev_W, prev_D) = A.shape
        filter_H, filter_W = self.params['kernel_shape']
        pad_h, pad_w = self.params['pad_h'], self.params['pad_w']

        dA = np.zeros((M, prev_H, prev_W, prev_D))
        self.grads = self.init_cache()

        A_pad = zero_pad(A, (pad_h, pad_w))
        dA_pad = zero_pad(dA, (pad_h, pad_w))

        for i in range(M):
            a_pad = A_pad[i]
            da_pad = dA_pad[i]

            for h in range(prev_H):
                for w in range(prev_W):
                    for c in range(self.params['filters']):
                        vert_start = self.params['stride']*h
                        vert_end = vert_start + filter_H
                        horiz_start = self.params['stride']*w
                        horiz_end = horiz_start + filter_W
                        a_slice = a_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        da_pad[vert_start:vert_end, horiz_start:horiz_end, :]+= self.params['W'][:,:,:,c] * dZ[i, h, w, c]
                        self.grads['dW'][:,:,:,c]+=a_slice*dZ[i, h, w,c]
                        self.grads['db'][:,:,:,c]+=dZ[i, h, w, c]
            dA[i, :, :, :] = da_pad[pad_h: -pad_h, pad_w: -pad_w, :]

        return dA

    def momentum(self, beta = 0.9):
        if not self.momentum_cache:
            self.momentum_cache = self.init_cache()
        self.momentum_cache['dW'] = beta * self.momentum_cache['dW'] + (1-beta) * self.grads['dW']
        self.momentum_cache['db'] = beta * self.momentum_cache['db'] + (1-beta) * self.grads['db']

    def rmsprop(self, beta=0.999, amsprop=True):
        if not self.rmsprop_cache:
            self.rmsprop_cache = self.init_cache()

        new_dW = beta * self.rmsprop_cache['dW'] + (1-beta) * (self.grads['dW']**2)
        new_db = beta * self.rmsprop_cache['db'] + (1-beta) * (self.grads['db']**2)

        if amsprop:
            self.rmsprop_cache['dW'] = np.maximum(self.rmsprop_cache['dW'], new_dW)
            self.rmsprop_cache['db'] = np.maximum(self.rmsprop_cache['db'], new_db)
        else:
            self.rmsprop_cache['dW'] = new_dW
            self.rmsprop_cache['db'] = new_db

    def apply_grads(self, lr=0.001, l2_penalty=1e-4, optimization='adam', epsilon=1e-8,
                    correct_bias=False, beta1=0.9, beta2=0.999, iter=999):
        if optimization!= 'adam':
            self.params['W'] -= lr * (self.grads['dW'] + l2_penalty * self.params['W'])
            self.params['b'] -= lr * (self.grads['db'] + l2_penalty * self.params['b'])

        else:
            if correct_bias:
                W_1st = self.momentum_cache['dW'] / (1 - beta1 ** iter)
                b_1st = self.momentum_cache['db'] / (1 - beta1 ** iter)
                W_2nd = self.momentum_cache['dW'] / (1 - beta2 ** iter)
                b_2nd = self.momentum_cache['db'] / (1 - beta2 ** iter)

                W_learning_rate = lr / (np.sqrt(W_2nd) + epsilon)
                b_learning_rate = lr / (np.sqrt(b_2nd) + epsilon)

                self.params['W'] -= W_learning_rate * (W_1st + l2_penalty * self.params['W'])
                self.params['b'] -= b_learning_rate * (b_1st + l2_penalty * self.params['b'])






