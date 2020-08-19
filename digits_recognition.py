from sklearn import datasets as sklearndatasets
import numpy
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    if x > 30:
        return 1.0
    elif x < -30:
        return 0.0
    return 1.0 / (1.0 + numpy.exp(-x))


def derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def bounds(x, left, right):
    if x < left:
        return left
    if x > right:
        return right
    return x


class DigitsRecognitionNeuralNetwork:
    def __init__(self):
        # Layers: 64 -> 32 -> 20 -> 10
        self.layer_sizes = [(64, 32), (32, 20), (20, 10)]
        self.weights = [2 * numpy.random.random_sample(layer_size) - 1
                        for layer_size in self.layer_sizes]

    def train_sample(self, data, learning_coef):
        (image, target) = data
        values = [numpy.divide(numpy.copy(image).flatten(), 15.0),
                  numpy.zeros(32),
                  numpy.zeros(20),
                  numpy.zeros(10)]

        for i in range(3):
            values[i+1] = numpy.dot(values[i], self.weights[i])
            values[i+1] = numpy.asarray([sigmoid(x) for x in values[i+1]])

        errors = [numpy.zeros(64),
                  numpy.zeros(32),
                  numpy.zeros(20),
                  numpy.zeros(10)]
        errors[3] = -values[3]
        errors[3][target] += 1

        for i in range(2, -1, -1):
            errors[i] = numpy.dot(errors[i+1], numpy.transpose(self.weights[i]))

        for i in range(3):
            n, m = self.layer_sizes[i]
            for x in range(n):
                for y in range(m):
                    self.weights[i][x][y] += learning_coef * errors[i+1][y] *\
                                             bounds(derivative(values[i+1][y]), 0.0001, 0.9999) * values[i][x]

        # print(self.weights)

    def check_sample(self, data):
        (image, target) = data
        values = [numpy.divide(numpy.copy(image).flatten(), 15.0),
                  numpy.zeros(32),
                  numpy.zeros(20),
                  numpy.zeros(10)]

        for i in range(3):
            values[i+1] = numpy.dot(values[i], self.weights[i])
            values[i+1] = numpy.asarray([sigmoid(x) for x in values[i+1]])

        errors = [numpy.zeros(64),
                  numpy.zeros(32),
                  numpy.zeros(20),
                  numpy.zeros(10)]

        errors[3] = -values[3]
        errors[3][target] += 1

        result = numpy.argmax(values[3])

        return result == target, numpy.sum(errors[3]**2)

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


def main():
    digits_dataset = sklearndatasets.load_digits()
    network = DigitsRecognitionNeuralNetwork()

    all_errors, all_pass_rates = [], []
    for iteration in range(30):
        error, pass_rate = network.check(digits_dataset)
        print("Iteration {0}: pass rate {1}, error {2}".format(iteration+1, pass_rate, error))
        all_errors.append(error)
        all_pass_rates.append(pass_rate)
        network.train(digits_dataset, 0.3, 0.1*(1-iteration/30))

    plt.figure(1)
    plt.plot(all_errors)
    plt.figure(2)
    plt.plot(all_pass_rates)
    plt.show()


if __name__ == "__main__":
    main()
