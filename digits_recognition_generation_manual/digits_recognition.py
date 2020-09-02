from sklearn import datasets as sklearndatasets
import matplotlib.pyplot as plt
import argparse
from digitsnn import DigitsRecognitionNeuralNetwork
from digitsnn import check_float_value_0_1, check_non_negative_int


def main():
    parser = argparse.ArgumentParser(description="Digits Recognition Neural Network")
    parser.add_argument("--iterations", type=check_non_negative_int, default=30,
                        help="NN learning iterations (default 30)")
    parser.add_argument("--check", action='store_true', help="Perform check of NN")
    parser.add_argument("--hide-plots", action='store_true', help="Do not show figures with ML stats")
    parser.add_argument("--nn-save-to", type=str, default=None,
                        help="If specified, save network after training to text file")
    parser.add_argument("--nn-load-from", type=str, default=None,
                        help="If specified, load network from text file instead of training")
    parser.add_argument("--learning-density", type=check_float_value_0_1, default=0.2,
                        help="How much dataset should be used in one iteration to learn (default 0.2)")
    parser.add_argument("--learning-coefficient", type=check_float_value_0_1, default=0.1,
                        help="Learning coefficient (default 0.1)")
    parser.add_argument("--disable-dynamic-learning-coefficient", action='store_true',
                        help="Disable decreasing learning coefficient linearly to zero at last iteration")
    parser.add_argument("--bias-neuron", action='store_true',
                        help="Add bias neuron to neural network (useful only for new network)")
    args = parser.parse_args()

    digits_dataset = sklearndatasets.load_digits()
    load_from_filename = args.nn_load_from
    if load_from_filename:
        network = DigitsRecognitionNeuralNetwork.load(load_from_filename)
    else:
        bias_neuron = args.bias_neuron
        network = DigitsRecognitionNeuralNetwork([64, 32, 20, 10], bias_neuron)

    all_errors, all_pass_rates = [], []
    called_train = False

    total_iterations = args.iterations
    learning_density = args.learning_density
    learning_coefficient = args.learning_coefficient
    disable_dynamic_learning_coefficient = args.disable_dynamic_learning_coefficient

    for iteration in range(total_iterations):
        pass_rate, error = network.check(digits_dataset)
        print("Learning: iteration {0}, pass rate: {1:.2f}%, error {2:.3f}"
              .format(iteration+1, pass_rate * 100.0, error))
        all_errors.append(error)
        all_pass_rates.append(pass_rate)
        if iteration + 1 != total_iterations:
            current_learning_coefficient = learning_coefficient
            if not disable_dynamic_learning_coefficient:
                current_learning_coefficient *= (1 - iteration / total_iterations)
            network.train(digits_dataset, learning_density, current_learning_coefficient)
            called_train = True
    else:
        if not args.hide_plots and total_iterations > 0:
            plt.figure(1)
            plt.plot(all_errors)
            plt.figure(2)
            plt.plot(all_pass_rates)
            plt.show()

    if args.check:
        pass_rate, error = network.check(digits_dataset)
        print("Checked, pass rate: {0:.2f}%, error {1:.3f}".format(pass_rate * 100.0, error))

    save_to_filename = args.nn_save_to
    if save_to_filename and called_train:
        network.save(save_to_filename)


if __name__ == "__main__":
    main()
