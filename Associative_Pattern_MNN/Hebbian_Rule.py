import numpy as np
import copy


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


def create_patterns(number_of_patterns, patterns, shape):
    patterns_array = []
    for i in range(number_of_patterns):
        patterns_array.append(create_new_input(shape, -1))
        create_character(patterns[i], patterns_array[i])
        # print_character(patterns_array[i])
        patterns_array[i] = patterns_array[i].flatten()
        patterns_array[i] = np.matrix(patterns_array[i])
    return patterns_array


def compare_inputs_outputs(inputs_array, outputs_array, weight_array):
    for i in range(len(inputs_array)):
        print("Pattern number ", i, end=" ")
        if np.all(np.sign(inputs_array[i] * weight_array) == outputs_array[i]):
            print("True")
        else:
            print("False")
    print()


def add_noise(percentage, patterns):
    output_with_noise = []
    for i in range(len(patterns)):
        temp = np.squeeze(np.asarray(copy.deepcopy(inputs[i])))
        random_choose = np.random.choice(len(temp), int((len(temp) * percentage) / 100), replace=False)
        for j in random_choose:
            temp[j] = -temp[j]
        output_with_noise.append(np.matrix(temp))
    return output_with_noise


def add_lose(percentage, patterns):
    output_with_lose = []
    for i in range(len(patterns)):
        temp = np.squeeze(np.asarray(copy.deepcopy(inputs[i])))
        random_choose = np.random.choice(len(temp), int((len(temp) * percentage) / 100), replace=False)
        for j in random_choose:
            temp[j] = 0
        output_with_lose.append(np.matrix(temp))
    return output_with_lose


def find_weight(inputs_matrix, outputs_matrix):
    weight_matrix = np.matrix(np.zeros(shape=(63, 15)))
    for i in range(len(inputs_matrix)):
        weight_matrix = inputs_matrix[i].transpose().dot(outputs_matrix[i]) + weight_matrix
    return weight_matrix


input_patterns = [([0, 3], [1, 3], [2, 2], [2, 4], [3, 2], [3, 4], [4, 2], [4, 3], [4, 4], [5, 1], [5, 5], [6, 1],
                   [6, 5], [7, 0], [7, 6], [8, 0], [8, 6]),
                  ([0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 5], [2, 0], [2, 6], [3, 0], [3, 5], [4, 0],
                   [4, 1], [4, 2], [4, 3], [4, 4], [5, 0], [5, 5], [6, 0], [6, 6], [7, 0], [7, 5], [8, 0], [8, 1],
                   [8, 2], [8, 3], [8, 4]),
                  ([0, 2], [0, 3], [0, 4], [1, 1], [1, 5], [2, 0], [2, 6], [3, 0], [4, 0], [5, 0], [6, 0], [6, 6],
                   [7, 1], [7, 5], [8, 2], [8, 3], [8, 4])]

output_patterns = [([0, 1], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 2], [4, 0], [4, 2]),
                   ([0, 0], [0, 1], [1, 0], [1, 2], [2, 0], [2, 1], [3, 0], [3, 2], [4, 0], [4, 1]),
                   ([0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2])]

inputs = create_patterns(3, input_patterns, (9, 7))
outputs = create_patterns(3, output_patterns, (5, 3))
weight = find_weight(inputs, outputs)

print("Finding outputs from inputs without noise and lose: ")
compare_inputs_outputs(inputs, outputs, weight)

print("Finding outputs from inputs with 20% noise: ")
inputs_with_noise_20 = add_noise(20, inputs)
compare_inputs_outputs(inputs_with_noise_20, outputs, weight)

print("Finding outputs from inputs with 40% noise: ")
inputs_with_noise_40 = add_noise(40, inputs)
compare_inputs_outputs(inputs_with_noise_40, outputs, weight)

print("Finding outputs from inputs with 20% lose: ")
inputs_with_lose_20 = add_lose(20, inputs)
compare_inputs_outputs(inputs_with_lose_20, outputs, weight)

print("Finding outputs from inputs with 40% lose: ")
inputs_with_lose_40 = add_lose(40, inputs)
compare_inputs_outputs(inputs_with_lose_40, outputs, weight)

print("Maximum noise for generating output: ")
for i in range(10):
    print("With ", i * 10, " Percent Noise: ")
    inputs_with_noise_40 = add_noise(10 * i, inputs)
    compare_inputs_outputs(inputs_with_noise_40, outputs, weight)
