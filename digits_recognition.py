from sklearn import datasets as sklearndatasets
import numpy
import random
import matplotlib.pyplot as plt
import argparse
import json


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


class DigitsRecognitionNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.interlayer_sizes = [(x, y) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.depth = len(self.layer_sizes)
        self.weights = [2 * numpy.random.random_sample(interlayer_size) - 1
                        for interlayer_size in self.interlayer_sizes]

    def calculate_values_and_errors_output_layer(self, image, target):
        values = [numpy.divide(numpy.copy(image).flatten(), 15.0)]
        for i, _ in enumerate(self.layer_sizes[1:]):
            values.append(sigmoid_array(numpy.dot(values[i], self.weights[i])))

        errors_output_layer = -values[-1]
        errors_output_layer[target] += 1.0
        return values, errors_output_layer

    def train_sample(self, data, learning_coef):
        (image, target) = data
        values, errors_output_layer = self.calculate_values_and_errors_output_layer(image, target)

        errors = [numpy.zeros(size) for size in self.layer_sizes[:-1]]
        errors.append(errors_output_layer)

        for i in range(self.depth-2, -1, -1):
            errors[i] = numpy.dot(errors[i+1], numpy.transpose(self.weights[i]))

        for i, (n, m) in enumerate(self.interlayer_sizes):
            for x in range(n):
                for y in range(m):
                    self.weights[i][x][y] += learning_coef * errors[i+1][y] *\
                                             bounds(derivative(values[i+1][y]), 0.0001, 0.9999) * values[i][x]

    def check_sample(self, data):
        (image, target) = data
        values, errors_output_layer = self.calculate_values_and_errors_output_layer(image, target)
        result = numpy.argmax(values[-1])
        return result == target, numpy.sum(errors_output_layer**2)

    def train(self, dataset, learning_density, learning_coef):
        actual_dataset = [(image, target) for image, target in zip(dataset.images, dataset.target)
                          if random.random() < learning_density]
        for data in actual_dataset:
            self.train_sample(data, learning_coef)

    def check(self, dataset):
        actual_dataset = [(image, target) for image, target in zip(dataset.images, dataset.target)]
        passed, total = 0, len(actual_dataset)
        error = 0.0
        for data in actual_dataset:
            success, sample_error = self.check_sample(data)
            error += sample_error
            if success:
                passed += 1

        return error, passed / total


class DigitsRecognitionNeuralNetworkSaver:
    def __init__(self, filename):
        self.filename = filename

    def save(self, network):
        data = dict(layer_sizes=network.layer_sizes,
                    weights=[weights_layer.tolist() for weights_layer in network.weights])
        with open(self.filename, "w") as f:
            json.dump(data, fp=f)


class DigitsRecognitionNeuralNetworkLoader:
    def __init__(self, filename):
        self.filename = filename

    def load(self):
        with open(self.filename, "r") as f:
            data = json.load(fp=f)

        if not data:
            return None

        network = DigitsRecognitionNeuralNetwork(data["layer_sizes"])
        network.weights = [numpy.asarray(weights_layer) for weights_layer in data["weights"]]
        return network


def main():
    parser = argparse.ArgumentParser(description="Digits Recognition Neural Network")
    parser.add_argument("--iterations", type=int, default=30, help="NN learning iterations")
    parser.add_argument("--hide-plots", action='store_true', help="Do not show figures with ML stats")
    parser.add_argument("--nn-save-to", type=str, default=None,
                        help="If specified, save network after training to text file")
    parser.add_argument("--nn-load-from", type=str, default=None,
                        help="If specified, load network from text file instead of training")
    args = parser.parse_args()

    digits_dataset = sklearndatasets.load_digits()
    load_from_filename = args.nn_load_from
    if load_from_filename:
        loader = DigitsRecognitionNeuralNetworkLoader(load_from_filename)
        network = loader.load()
        error, pass_rate = network.check(digits_dataset)
        print("Pass rate {0}, error {1}".format(pass_rate, error))
    else:
        network = DigitsRecognitionNeuralNetwork([64, 32, 20, 10])
        all_errors, all_pass_rates = [], []
        total_iterations = args.iterations
        for iteration in range(total_iterations):
            error, pass_rate = network.check(digits_dataset)
            print("Iteration {0}: pass rate {1}, error {2}".format(iteration+1, pass_rate, error))
            all_errors.append(error)
            all_pass_rates.append(pass_rate)
            network.train(digits_dataset, 0.3, 0.1 * (1 - iteration/total_iterations))

        if not args.hide_plots:
            plt.figure(1)
            plt.plot(all_errors)
            plt.figure(2)
            plt.plot(all_pass_rates)
            plt.show()

        save_to_filename = args.nn_save_to
        if save_to_filename:
            saver = DigitsRecognitionNeuralNetworkSaver(save_to_filename)
            saver.save(network)


if __name__ == "__main__":
    main()
