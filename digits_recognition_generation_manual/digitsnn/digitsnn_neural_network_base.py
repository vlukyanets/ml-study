import numpy
import json
from digitsnn.utils import bounds, derivative


# Bias neuron is last in layer
class DigitsNeuralNetworkBase:
    def __init__(self, layer_sizes, has_bias_neuron=False):
        self.has_bias_neuron = has_bias_neuron
        self.layer_sizes = layer_sizes
        self.interlayer_sizes = [(x+int(self.has_bias_neuron), y)
                                 for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.depth = len(self.layer_sizes)
        self.weights = [2.0*numpy.random.random_sample(interlayer_size)-1
                        for interlayer_size in self.interlayer_sizes]

    def backward_propagation(self, learning_coefficient, values, error_output_layer):
        errors = [numpy.zeros(0) for _ in range(self.depth-1)]
        errors.append(numpy.append(error_output_layer, [0.0]))

        for i in range(self.depth-2, -1, -1):
            if self.has_bias_neuron:
                errors[i] = numpy.dot(errors[i+1][:-1], numpy.transpose(self.weights[i]))
            else:
                errors[i] = numpy.dot(errors[i+1], numpy.transpose(self.weights[i]))

        for i, (n, m) in enumerate(self.interlayer_sizes):
            for x in range(n):
                for y in range(m):
                    self.weights[i][x][y] += learning_coefficient * errors[i+1][y] * values[i][x] *\
                                             bounds(derivative(values[i+1][y]), 0.0001, 0.9999)

    def prepare_data_to_save(self):
        return dict(layer_sizes=self.layer_sizes,
                    weights=[weights_layer.tolist() for weights_layer in self.weights],
                    network_model=self.__class__.__name__)

    def save(self, filename):
        data = self.prepare_data_to_save()
        with open(filename, "w") as f:
            json.dump(data, fp=f)
