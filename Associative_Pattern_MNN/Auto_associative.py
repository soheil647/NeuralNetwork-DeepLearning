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


s = []
s0 = np.matrix([1, 1, 1, -1])
s.append(s0)
weight = find_weight(s, s)
print(weight)
compare_inputs_outputs(s, s, weight)

# print("2 vectors are not orthogonal so we cant save it")
# s1 = np.matrix([-1, 1, 1, -1])
# s.append(s1)
# weight = find_weight(s, s)
# print(weight)
# compare_inputs_outputs(s, s, weight)

s1 = np.matrix([1, -1, 1, 1])
s.append(s1)
weight = find_weight(s, s)
print(weight)
compare_inputs_outputs(s, s, weight)

s2 = np.matrix([-1, 1, 1, 1])
s.append(s2)
weight = find_weight(s, s)
print(weight)
compare_inputs_outputs(s, s, weight)

s3 = np.matrix([1, 1, -1, 1])
s.append(s3)
weight = find_weight(s, s)
print(weight)
compare_inputs_outputs(s, s, weight)