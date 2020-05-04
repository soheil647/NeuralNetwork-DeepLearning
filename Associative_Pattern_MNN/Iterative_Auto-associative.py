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
        # print(inputs_array[i])
        # print(outputs_array, "\n\n")
        print("Pattern number ", i, end=" ")
        if np.all(np.sign(inputs_array[i] * weight_array) == outputs_array):
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


def noise_3_value():
    pattern = []
    pattern.append(np.matrix([1, -1, -1, 1]))
    pattern.append(np.matrix([-1, 1, -1, 1]))
    pattern.append(np.matrix([-1, -1, 1, 1]))
    pattern.append(np.matrix([-1, -1, -1, -1]))
    return pattern


def iterative_net(inputs_pattern, tests_pattern):
    weight = find_weight(inputs_pattern, inputs_pattern)
    print(weight, "\n")
    last_result = []

    for i in range(len(tests_pattern)):
        result_patterns = []
        new_pattern = np.sign(tests_pattern[i] * weight)
        while not np.all(new_pattern == inputs_pattern):
            result_patterns.append(np.array(new_pattern).reshape(4, ))
            new_pattern = np.sign(new_pattern * weight)
            if check_repeated_results(result_patterns, new_pattern):
                print("Repeated pattern for Pattern: ", i)
                break
        last_result.append(np.array(new_pattern).reshape(4, ))
    print(last_result)
    compare_inputs_outputs(np.array(last_result), inputs_pattern, weight)


def check_repeated_results(results, new):
    for i in range(len(results)):
        if np.all(np.array(results) == np.array(new).reshape(4, )):
            return True
    return False


def create_random_index(indx):
    x = np.arange(indx)
    np.random.shuffle(x)
    return x


def hopfeild_net(inputs_pattern, tests_pattern):
    weight = find_weight(inputs_pattern, inputs_pattern)
    weight = np.array(weight)
    print(weight, "\n")
    last_result = []
    for i in range(len(tests_pattern)):
        x = np.array(tests_pattern[i]).reshape(4, )
        y = x
        indx = create_random_index(len(y))
        for j in indx:
            y[j] = np.sign(x[j] + np.sum(y * weight.transpose()[j]))
        last_result.append(y)
    print(last_result)
    compare_inputs_outputs(np.matrix(last_result), inputs_pattern, weight)


s = []
s0 = [1, 1, 1, -1]
s.append(np.matrix([1, 1, 1, -1]))

# iterative_net(np.matrix(s0), lose_3_value())
# iterative_net(np.matrix(s0), noise_3_value())

hopfeild_net(np.matrix(s0), lose_3_value())
hopfeild_net(np.matrix(s0), noise_3_value())
