import numpy as np

class CategoricalCrossEntropy:
    @staticmethod
    def compute_loss(labels, predictions, epsilon=1e-8):
        predictions /= np.sum(predictions, axis=0, keepdims=True)
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        return -np.sum(labels*np.log(predictions))

    @staticmethod
    def compute_derivative(labels, predictions):
        return labels-predictions