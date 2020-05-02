import numpy as np


def find_weight(inputs_matrix, outputs_matrix):
    weight_matrix = np.matrix(np.zeros(shape=(4, 4)))
    for i in range(len(inputs_matrix)):
        weight_matrix = inputs_matrix[i].transpose().dot(outputs_matrix[i]) + weight_matrix
    di = np.diag_indices(4)
    weight_matrix[di] = 0
    return weight_matrix


def compare_inputs_outputs(inputs_array, outputs_array, weight_array):
    for i in range(len(inputs_array)):
        print("Pattern number ", i, end=" ")
        if np.all(np.sign(inputs_array[i] * weight_array) == outputs_array[i]):
            print("True")
        else:
            print("False")
    print()


def lose_3_value():
    pattern = []
    pattern.append(np.matrix([0, 0, 0, -1]))
    pattern.append(np.matrix([0, 0, 1, 0]))
    pattern.append(np.matrix([0, 1, 0, 0]))
    pattern.append(np.matrix([1, 0, 0, 0]))
    return pattern


def iterative_net(inputs_pattern, tests_pattern):
    weight = find_weight(inputs_pattern, inputs_pattern)
    compare_inputs_outputs(inputs_pattern, inputs_pattern, weight)
    print(weight)

    for i in range(len(tests_pattern)):
        result_patterns = []
        result_patterns.append([0, 0, 0, 0])
        result_patterns.append([0, 0, 1, 0])
        new_pattern = tests_pattern[i] * weight
        print(new_pattern, "\n")
        while not np.all(new_pattern == inputs_pattern):
            result_patterns.append(np.array(new_pattern).reshape(4,))
            new_pattern = np.sign(new_pattern * weight)
            print(np.array(new_pattern).reshape(4,))
            print(np.array(result_patterns))
            print(np.array(new_pattern).reshape(4,) in np.array(result_patterns))
            if np.array(new_pattern).reshape(4,) in np.array(result_patterns):
                print("Repeated pattern")
                break
        print("\n")

s = []
s0 = [1, 1, 1, -1]
s.append(np.matrix([1, 1, 1, -1]))

iterative_net(np.matrix(s0), lose_3_value())
