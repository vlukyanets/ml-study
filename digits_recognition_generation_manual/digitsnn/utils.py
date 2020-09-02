import argparse
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


def check_float_value_0_1(value):
    fvalue = float(value)
    if fvalue <= 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError("Value should be in range (0; 1)")

    return fvalue


def check_int_value_0_9(value):
    ivalue = int(value)
    if ivalue < 0 or ivalue > 9:
        raise argparse.ArgumentTypeError("Value should be in range [0; 9]")

    return ivalue


def check_non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("Value count should not be negative")

    return ivalue
