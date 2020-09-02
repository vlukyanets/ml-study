from sklearn import datasets as sklearndatasets
import argparse
from digitsnn import DigitsGenerationNeuralNetwork, display_digit
from digitsnn import check_non_negative_int, check_int_value_0_9, check_float_value_0_1


def main():
    parser = argparse.ArgumentParser(description="Digits Generation Neural Network")
    parser.add_argument("--iterations", type=check_non_negative_int, default=30,
                        help="NN learning iterations (default 30)")
    parser.add_argument("--draw", type=check_int_value_0_9, help="Perform check of NN")
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
    args = parser.parse_args()

    digits_dataset = sklearndatasets.load_digits()
    load_from_filename = args.nn_load_from
    if load_from_filename:
        network = DigitsGenerationNeuralNetwork.load(load_from_filename)
    else:
        network = DigitsGenerationNeuralNetwork([10, 20, 32, 64])

    all_errors = []
    total_iterations = args.iterations
    learning_density = args.learning_density
    learning_coefficient = args.learning_coefficient
    disable_dynamic_learning_coefficient = args.disable_dynamic_learning_coefficient
    called_train = False

    for iteration in range(total_iterations):
        error = network.check_generation(digits_dataset)
        print("Learning: iteration {0}, error {1}".format(iteration+1, error))
        all_errors.append(error)
        if iteration + 1 != total_iterations:
            current_learning_coefficient = learning_coefficient
            if not disable_dynamic_learning_coefficient:
                current_learning_coefficient *= (1 - iteration / total_iterations)
            network.train_generation(digits_dataset, learning_density, current_learning_coefficient)
            called_train = True

    save_to_filename = args.nn_save_to
    if save_to_filename and called_train:
        network.save(save_to_filename)

    digit = args.draw
    image = network.draw(digit)
    display_digit(image)


if __name__ == "__main__":
    main()
