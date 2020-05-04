import random

import numpy as np

x_layer = [([0, 7], [0, 8], [1, 7], [1, 8], [2, 7], [2, 8], [3, 7], [3, 8], [4, 7], [4, 8], [5, 6], [5, 7], [5, 8],
            [5, 9], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9],
            [7, 10], [7, 11], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [9, 2],
            [9, 3], [9, 4], [9, 7], [9, 8], [9, 11], [9, 12], [9, 13], [10, 1], [10, 2], [10, 3], [10, 7], [10, 8],
            [10, 12], [10, 13], [10, 14], [11, 0], [11, 1], [11, 2], [11, 7], [11, 8], [11, 13], [11, 14], [11, 15],
            [12, 0], [12, 1], [12, 7], [12, 8], [12, 14], [12, 15], [13, 0], [13, 7], [13, 8], [13, 15], [14, 7],
            [14, 8], [15, 6], [15, 7], [15, 8], [15, 9], [16, 6], [16, 7], [16, 8], [16, 9], [17, 6], [17, 9]),
           ([0, 6], [0, 7], [0, 8], [0, 9], [1, 7], [1, 8], [2, 7], [2, 8], [3, 7], [3, 8], [4, 7], [4, 8], [5, 7],
            [5, 8], [6, 7], [6, 8], [7, 7], [7, 8], [8, 7], [8, 8], [9, 2], [9, 3], [9, 7], [9, 8], [9, 12], [9, 13],
            [10, 2], [10, 3], [10, 6], [10, 7], [10, 8], [10, 9], [10, 12], [10, 13], [11, 2], [11, 3], [11, 5],
            [11, 6],
            [11, 7], [11, 8], [11, 9], [11, 10], [11, 12], [11, 13], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6],
            [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [13, 2], [13, 3], [13, 4], [13, 5],
            [13, 6], [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [14, 2], [14, 3], [14, 5],
            [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [14, 12], [14, 13], [15, 2], [15, 3], [15, 6], [15, 7],
            [15, 8], [15, 9], [15, 12], [15, 13], [16, 2], [16, 3], [16, 12], [16, 13], [17, 2], [17, 3], [17, 12],
            [17, 13])]

y_layer = [([0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 7], [0, 16], [0, 17], [0, 21], [0, 26], [0, 28], [0, 29],
            [0, 31], [0, 32], [0, 33], [1, 0], [1, 5], [1, 7], [1, 15], [1, 18], [1, 21], [1, 22], [1, 26], [1, 28],
            [2, 0], [2, 5], [2, 7], [2, 15], [2, 18], [2, 21], [2, 22], [2, 26], [2, 28], [3, 0], [3, 5], [3, 7],
            [3, 15], [3, 18], [3, 21], [3, 23], [3, 26], [3, 28], [3, 29], [3, 30], [3, 31], [3, 32], [4, 0], [4, 1],
            [4, 2], [4, 3], [4, 4], [4, 7], [4, 14], [4, 19], [4, 21], [4, 24], [4, 26], [4, 28], [5, 0], [5, 7],
            [5, 14], [5, 15], [5, 16], [5, 17], [5, 18], [5, 19], [5, 21], [5, 25], [5, 26], [5, 28], [6, 0], [6, 7],
            [6, 12], [6, 14], [6, 19], [6, 21], [6, 25], [6, 26], [6, 28], [7, 0], [7, 7], [7, 8], [7, 9], [7, 10],
            [7, 11], [7, 12], [7, 14], [7, 19], [7, 21], [7, 26], [7, 28], [7, 29], [7, 30], [7, 31], [7, 32],
            [7, 33]),
           ([0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 12], [0, 18], [0, 25], [0, 28], [0, 33],
            [1, 3], [1, 11], [1, 13], [1, 18], [1, 19], [1, 25], [1, 28], [1, 32], [2, 3], [2, 10], [2, 14], [2, 18],
            [2, 20], [2, 25], [2, 28], [2, 31], [3, 3], [3, 10], [3, 14], [3, 18], [3, 21], [3, 25], [3, 28], [3, 29],
            [3, 30], [4, 3], [4, 9], [4, 15], [4, 18], [4, 22], [4, 25], [4, 28], [4, 31], [5, 3], [5, 9], [5, 10],
            [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [5, 18], [5, 23], [5, 25], [5, 28], [5, 32], [6, 3], [6, 9],
            [6, 15], [6, 18], [6, 24], [6, 25], [6, 28], [6, 33], [7, 3], [7, 9], [7, 15], [7, 18], [7, 25], [7, 28],
            [7, 34])]


def compare_inputs_outputs_x_to_y(inputs_array, outputs_array, weight_array):
    for i in range(len(inputs_array)):
        print("Pattern number ", i, end=" ")
        if np.all(np.sign(inputs_array[i] * weight_array) == outputs_array[i]):
            print("True")
        else:
            print("False")
    print()


def compare_inputs_outputs_y_to_x(inputs_array, outputs_array, weight_array):
    for i in range(len(inputs_array)):
        print("Pattern number ", i, end=" ")
        if np.all(np.sign(np.matrix(outputs_array[i]) * np.array(weight_array).transpose()) == inputs_array[i]):
            print("True")
        else:
            print("False")
    print()


def create_new_input(shape, value):
    array = np.empty(shape)
    array.fill(value)
    return array


def create_character(values, array):
    for i, j in values:
        array[i][j] = 1


def print_character(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i, j] == 1:
                print("#", end=' ')
            else:
                print(".", end=' ')
        print()
    print()


def show_character(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] == 1:
                print("#", end=' ')
            else:
                print(".", end=' ')
        print()
    print()


def create_patterns(number_of_patterns, patterns, shape):
    patterns_array = []
    for i in range(number_of_patterns):
        patterns_array.append(create_new_input(shape, -1))
        create_character(patterns[i], patterns_array[i])
        # print_character(patterns_array[i])
        patterns_array[i] = patterns_array[i].flatten()
    return patterns_array


def find_weight(inputs_matrix, outputs_matrix):
    weight_matrix = np.array(np.zeros(shape=(288, 280)))
    for i in range(len(inputs_matrix)):
        weight_matrix = np.matrix(inputs_matrix[i]).transpose().dot(np.matrix(outputs_matrix[i])) + np.matrix(weight_matrix)
    return weight_matrix


def bam_net(test_patterns_x, test_patterns_y):
    inputs = create_patterns(2, x_layer, (18, 16))
    outputs = create_patterns(2, y_layer, (8, 35))
    weight = find_weight(inputs, outputs)
    print(weight)
    compare_inputs_outputs_x_to_y(inputs, outputs, weight)
    compare_inputs_outputs_y_to_x(inputs, outputs, weight)
    last_result_y = []
    last_result_x = []
    test_patterns_x = create_patterns(2, test_patterns_x, (18, 16))
    test_patterns_y = create_patterns(2, test_patterns_y, (8, 35))
    for i in range(len(test_patterns_x)):
        x = test_patterns_x[i]
        y = test_patterns_y[i]

        for j in range(len(test_patterns_y)):
            t = y
            y[j] = np.sign(np.sum(x * np.array(weight).transpose()[j]))
            print(np.all(t == y))
        last_result_y.append(y)

        for j in range(len(test_patterns_x)):
            x[j] = np.sign(np.sum(np.matrix(y) * weight[j].transpose()))
        last_result_x.append(x)

    compare_inputs_outputs_x_to_y(last_result_x, outputs, weight)
    compare_inputs_outputs_y_to_x(inputs, last_result_y, weight)
    # show_character(np.array(last_result_x[0]).reshape(18, 16))
    # show_character(np.array(last_result_x[1]).reshape(18, 16))
    # show_character(np.array(last_result_y[0]).reshape(8, 35))
    # show_character(np.array(last_result_y[1]).reshape(8, 35))


def noise_inputs_oututs(inputs_pattern, outputs_pattern, percent):
    input_noise_pattern = []
    output_noise_pattern = []
    for i in range(len(inputs_pattern)):
        random_elements_input = random.sample(inputs_pattern[i], k=int(len(inputs_pattern[i])*percent))
        random_elements_output = random.sample(outputs_pattern[i], k=int(len(outputs_pattern[i])*percent))
        input_noise_pattern.append(tuple(x for x in inputs_pattern[i] if x not in random_elements_input))
        output_noise_pattern.append(tuple(x for x in outputs_pattern[i] if x not in random_elements_output))
    return input_noise_pattern, output_noise_pattern


noise_input, noise_output = noise_inputs_oututs(x_layer, y_layer, 0.1)
bam_net(noise_input, noise_output)