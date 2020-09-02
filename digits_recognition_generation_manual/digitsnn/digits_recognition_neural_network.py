import json
import numpy
import random
from digitsnn.digitsnn_neural_network_base import DigitsNeuralNetworkBase
from digitsnn.utils import sigmoid_array


class DigitsRecognitionNeuralNetwork(DigitsNeuralNetworkBase):
    def __init__(self, layer_sizes, has_bias_neuron):
        super().__init__(layer_sizes, has_bias_neuron)

    def forward_propagation(self, input_data):
        (image, target) = input_data
        values = [numpy.copy(image).flatten()]
        for i in range(0, self.depth-1):
            if self.has_bias_neuron:
                values[-1] = numpy.append(values[-1], [1.0])
            values.append(sigmoid_array(numpy.dot(values[i], self.weights[i])))

        errors_output_layer = -values[-1]
        errors_output_layer[target] += 1.0
        return values, errors_output_layer

    def train_sample(self, data, learning_coefficient):
        (image, target) = data
        values, errors_output_layer = self.forward_propagation((numpy.divide(image, 15.0), target))
        self.backward_propagation(learning_coefficient, values, errors_output_layer)

    def check_sample(self, data):
        (image, target) = data
        values, errors_output_layer = self.forward_propagation((numpy.divide(image, 15.0), target))
        result = numpy.argmax(values[-1])
        return result == target, numpy.sum(errors_output_layer**2)

    def train(self, dataset, learning_density, learning_coefficient):
        actual_dataset = [(image, target) for image, target in zip(dataset.images, dataset.target)
                          if random.random() <= learning_density]
        for data in actual_dataset:
            self.train_sample(data, learning_coefficient)

    def check(self, dataset):
        actual_dataset = [(image, target) for image, target in zip(dataset.images, dataset.target)]
        error, passed, total = 0.0, 0, len(actual_dataset)
        for data in actual_dataset:
            success, sample_error = self.check_sample(data)
            error += sample_error
            if success:
                passed += 1

        return passed / total, error

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            data = json.load(fp=f)

        current_network_model = data["network_model"]
        if current_network_model != DigitsRecognitionNeuralNetwork.__name__:
            raise TypeError("Network model is not {0}".format(DigitsRecognitionNeuralNetwork.__name__))

        layer_sizes, weights = data["layer_sizes"], data["weights"]
        network = DigitsRecognitionNeuralNetwork(layer_sizes)
        network.weights = [numpy.asarray(weights_layer) for weights_layer in weights]
        return network
