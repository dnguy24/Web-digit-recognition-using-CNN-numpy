import numpy as np
import gzip
def zero_pad(X, pad):
    return np.pad(X, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), 'constant')

def get_fans(shape):
    '''
    :param shape:
    :return:
    '''
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

def uniform(shape, scale=0.05):
    '''
    :param shape:
    :param scale:
    :return:
    '''
    return np.random.uniform(-scale, scale, size=shape)

def glorot_normal(shape):
    '''
    A function for smart uniform distribution based initialization of parameters
    [Glorot et al. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
    '''
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(2. / (fan_in + fan_out))
    shape = (fan_out, fan_in) if len(shape) == 2 else shape  # For a fully connected network
    bias_shape = (fan_out, 1) if len(shape) == 2 else (
        1, 1, 1, shape[3])  # This supports only CNNs and fully connected networks
    return random_initializer(shape, scale), uniform(shape=bias_shape)

def glorot_uniform(shape):
    '''
    A function for smart uniform distribution based initialization of parameters
    [Glorot et al. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
    '''
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(6. / (fan_in + fan_out))
    shape = (fan_out, fan_in) if len(shape) == 2 else shape  # For a fully connected network
    bias_shape = (fan_out, 1) if len(shape) == 2 else (
        1, 1, 1, shape[3])  # This supports only CNNs and fully connected networks
    return uniform(shape, scale), uniform(shape=bias_shape)

def random_initializer(shape, scale=0.05):
    return np.random.normal(0, scale, size=shape)

def random_ini(shape, scale=0.05):
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(6. / (fan_in + fan_out))
    shape = (fan_out, fan_in) if len(shape) == 2 else shape  # For a fully connected network
    bias_shape = (fan_out, 1) if len(shape) == 2 else (
        1, 1, 1, shape[3])  # This supports only CNNs and fully connected networks
    return np.random.normal(0, scale, size=shape), np.random.normal(0, scale, size=bias_shape)

def calibrated_initializer(shape, scale=0.05):
    n, _ = get_fans(shape)
    return np.random.normal(0, scale, size=shape) * np.sqrt(2.0/n), np.random.normal(0, scale, size=shape) * np.sqrt(2.0/n)

def get_batches(data, labels=None, batch_size=256, shuffle=True):
    '''
    Function to get data in batches.
    :param data:[numpy array]: training or test data. Assumes shape=[M, N] where M is the features and N is samples.
    :param labels:[numpy array, Default = None (for without labels)]: actual labels corresponding to the data.
    Assumes shape=[M, N] where M is number of classes/results per sample and N is number of samples.
    :param batch_size:[int, Default = 256]: required size of batch. If data can't be exactly divided by batch_size,
    remaining samples will be in a new batch
    :param shuffle:[boolean, Default = True]: if true, function will shuffle the data
    :return:[numpy array, numpy array]: batch data and corresponding labels
    '''
    N = data.shape[1] if len(data.shape) == 2 else data.shape[0]
    num_batches = N//batch_size
    if len(data.shape) == 2:
        data = data.T
    if shuffle:
        shuffled_indices = np.random.permutation(N)
        data = data[shuffled_indices]
        labels = labels[:, shuffled_indices] if labels is not None else None
    if num_batches == 0:
        if labels is not None:
            yield (data.T, labels) if len(data.shape) == 2 else (data, labels)
        else:
            yield data.T if len(data.shape) == 2 else data
    for batch_num in range(num_batches):
        if labels is not None:
            yield (data[batch_num*batch_size:(batch_num+1)*batch_size].T,
                  labels[:, batch_num*batch_size:(batch_num+1)*batch_size]) if len(data.shape) == 2 \
                      else (data[batch_num*batch_size:(batch_num+1)*batch_size],
                  labels[:, batch_num*batch_size:(batch_num+1)*batch_size])
        else:
            yield data[batch_num*batch_size:(batch_num+1)*batch_size].T if len(data.shape) == 2 else \
                data[batch_num*batch_size:(batch_num+1)*batch_size]
    if N%batch_size != 0 and num_batches != 0:
        if labels is not None:
            yield (data[num_batches*batch_size:].T, labels[:, num_batches*batch_size:]) if len(data.shape) == 2 else \
                (data[num_batches*batch_size:], labels[:, num_batches*batch_size:])
        else:
            yield data[num_batches*batch_size:].T if len(data.shape)==2 else data[num_batches*batch_size:]


def evaluate(labels, predictions):
    '''
    A function to compute the accuracy of the predictions on a scale of 0-1.
    :param labels:[numpy array]: Training labels (or testing/validation if available)
    :param predictions:[numpy array]: Predicted labels
    :return:[float]: a number between [0, 1] denoting the accuracy of the prediction
    '''
    return np.mean(np.argmax(labels, axis=0) == np.argmax(predictions, axis=0))
