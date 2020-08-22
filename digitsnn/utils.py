import numpy


def sigmoid(x: float):
    if x > 50:
        return 1.0
    elif x < -50:
        return 0.0
    return 1.0 / (1.0 + numpy.exp(-x))


def sigmoid_array(x):
    return numpy.asarray([sigmoid(z) for z in x])


def derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def bounds(x, left, right):
    if x < left:
        return left
    if x > right:
        return right
    return x
