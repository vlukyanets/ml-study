import json
import numpy
import random
from digitsnn.digitsnn_neural_network_base import DigitsNeuralNetworkBase
from digitsnn.utils import sigmoid_array


class DigitsGenerationNeuralNetwork(DigitsNeuralNetworkBase):
    def __init__(self, layer_sizes):
        super().__init__(layer_sizes)

    def forward_propagation(self, input_data, output_to_compare_error=None):
        values = [input_data]
        for i in range(0, self.depth - 1):
            values.append(sigmoid_array(numpy.dot(values[i], self.weights[i])))

        return values,\
               numpy.subtract(output_to_compare_error, values[-1]) if output_to_compare_error is not None else None

    def train_generation_sample(self, learning_coefficient, target, image):
        input_data = numpy.zeros(self.layer_sizes[0])
        input_data[target] = 1.0
        values, errors_output_layer = self.forward_propagation(input_data,
                                                               numpy.divide(numpy.copy(image), 15.0).flatten())
        self.backward_propagation(learning_coefficient, values, errors_output_layer)

    def check_generation_sample(self, target, image):
        input_data = numpy.zeros(self.layer_sizes[0])
        input_data[target] = 1.0
        _, error = self.forward_propagation(input_data, numpy.divide(numpy.copy(image), 15.0).flatten())
        return numpy.sum(error ** 2)

    def train_generation(self, dataset, learning_density, learning_coef):
        actual_dataset = [(image, target) for image, target in zip(dataset.images, dataset.target)
                          if random.random() <= learning_density]
        for image, target in actual_dataset:
            self.train_generation_sample(learning_coef, target, image)

    def check_generation(self, dataset):
        actual_dataset = [(image, target) for image, target in zip(dataset.images, dataset.target)]
        error = 0.0
        for image, target in actual_dataset:
            sample_error = self.check_generation_sample(target, image)
            error += sample_error

        return error

    def calculate(self, target):
        input_data = numpy.zeros(10)
        input_data[target] = 1.0
        values, _ = self.forward_propagation(input_data, None)
        return values[-1]

    def draw(self, target):
        output_layer = self.calculate(target)
        output_layer = numpy.array([int(round(x * 15.0)) for x in output_layer])
        return output_layer.reshape((8, 8)).tolist()

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            data = json.load(fp=f)

        current_network_model = data["network_model"]
        if current_network_model != DigitsGenerationNeuralNetwork.__name__:
            raise TypeError("Network model is not {0}".format(DigitsGenerationNeuralNetwork.__name__))

        layer_sizes, weights = data["layer_sizes"], data["weights"]
        network = DigitsGenerationNeuralNetwork(layer_sizes)
        network.weights = [numpy.asarray(weights_layer) for weights_layer in weights]
        return network
