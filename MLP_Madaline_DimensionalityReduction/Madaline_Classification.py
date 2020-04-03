import numpy as np
import matplotlib.pyplot as plt

Orange_x1 = np.random.normal(3, 0.2, 50)
Orange_x2 = np.random.normal(2.5, 0.3, 50)

Green_x1 = np.random.normal(1, 0.2, 50)
Green_x2 = np.random.normal(2, 0.2, 50)

Blue_x1 = [0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
Blue_x2 = [2, 1, 3.5, 1, 2, 3, 1, 4, 2, 4]

plt.plot(Orange_x1, Orange_x2, "bo")
plt.plot(Green_x1, Green_x2, "ro")
plt.plot(Blue_x1, Blue_x2, "yo")

plt.grid()
plt.show()
dataset_list = []
for i in range(len(Orange_x1)):
    dataset_list.append([Orange_x1[i], Orange_x2[i], [-1, 1]])
for i in range(len(Green_x1)):
    dataset_list.append([Green_x1[i], Green_x2[i], [1, -1]])
for i in range(len(Blue_x1)):
    dataset_list.append([Blue_x1[i], Blue_x2[i], [-1, -1]])

# print(np.array(dataset_list).reshape(110, 3))


# Make a prediction with weights
def predict_step(activation):
    return 1.0 if activation >= 0.0 else -1.0


def find_net(in_data, out_weight, out_bias):
    print(out_bias)
    activation = out_bias
    for j in range(len(in_data) - 1):
        activation += out_weight[j] * in_data[j]
    return activation


# Estimate Perceptron weights using stochastic gradient descent
def train_weights_madaline_mr1(x_in, number_of_layer1_neuron, number_of_and_input, l_rate, n_epoch):
    input_weights_vector = [[0.0 for i in range(len(x_in[0]) - 1)] for j in range(number_of_layer1_neuron)]
    input_bias_vector = [0.0 for i in range(number_of_layer1_neuron)]
    and1_weights_vector = [[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]]
    and1_bias_vector = [-2, -3]

    predict_layer1_value = [0.0 for i in range(number_of_layer1_neuron)]
    predict_and_value = [0.0 for i in range(number_of_and_input)]

    sum_error = 0
    for i_epoch in range(n_epoch):
        for data in x_in:
            predict_and_value = []
            activation_layer1 = []
            for neuron_layer1_number in range(number_of_layer1_neuron):

                activation_layer1[neuron_layer1_number] = find_net(data,
                                                                   input_weights_vector[neuron_layer1_number],
                                                                   input_bias_vector[neuron_layer1_number])
                predict_layer1_value[neuron_layer1_number] = predict_step(activation_layer1)

            for and_number in range(number_of_and_input):
                activation_and = find_net(predict_layer1_value,
                                          and1_weights_vector[and_number],
                                          and1_bias_vector[and_number])
                predict_and_value[and_number] = predict_step(activation_and)

                error = x_in[2][and_number] - predict_and_value[and_number]
                if error != 0 and x_in[2][and_number] == 1:
                    for neuron_layer1_number in range(number_of_layer1_neuron):
                        if predict_layer1_value[neuron_layer1_number] < 0:
                            input_bias_vector[neuron_layer1_number] = input_bias_vector[neuron_layer1_number] + l_rate * (activation_layer1[number_of_layer1_neuron] - x_in[2][and_number])
                            for input_number in range(data):
                                input_weights_vector[neuron_layer1_number][input_number] = input_weights_vector[neuron_layer1_number] + l_rate * (activation_layer1[number_of_layer1_neuron] - x_in[2][and_number]) * data[input_number]

                if error != 0 and x_in[2][and_number] == -1:
                    for neuron_layer1_number in range(number_of_layer1_neuron):
                        if predict_layer1_value[neuron_layer1_number] > 0:
                            input_bias_vector[neuron_layer1_number] = input_bias_vector[neuron_layer1_number] + l_rate * (activation_layer1[number_of_layer1_neuron] - x_in[2][and_number])
                            for input_number in range(data):
                                input_weights_vector[neuron_layer1_number][input_number] = input_weights_vector[neuron_layer1_number] + l_rate * (activation_layer1[number_of_layer1_neuron] - x_in[2][and_number]) * data[input_number]
                sum_error += error ** 2
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (i_epoch, l_rate, sum_error))
    return input_weights_vector, input_bias_vector


epoch = 4
weights, bias = train_weights_madaline_mr1(x_in=dataset_list, number_of_layer1_neuron=7, number_of_and_input=2, l_rate=0.2, n_epoch=epoch)
print(weights[0])
print(bias)
# ###############          Enter your code above ...           ##################
#
print("\nThe Neural Network has been trained in " + str(epoch) + " epochs.")
#
# ###############                   Testing                    ##################