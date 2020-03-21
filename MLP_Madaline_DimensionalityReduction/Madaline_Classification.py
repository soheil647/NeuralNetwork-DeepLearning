import numpy as np

dataset = np.array(read_train_file())


# Make a prediction with weights
def predict_step(activation):
    return 1.0 if activation >= 0.0 else -1.0


def find_net(in_data, out_weight, out_bias):
    activation = out_bias
    for j in range(len(in_data)):
        activation += out_weight[j] * in_data[j]
    return activation


# Estimate Perceptron weights using stochastic gradient descent
def train_weights_MAdaline_MR1(x_in, target, number_of_layer1_neuron, number_of_and_input, l_rate, n_epoch):
    input_weights_vector = [[0.0 for i in range(len(x_in[0]))] for j in range(len(number_of_layer1_neuron))]
    input_bias_vector = [0.0 for i in range(number_of_layer1_neuron)]

    and1_weights_vector = [[0.0 for i in range(len(number_of_layer1_neuron))] for j in range(len(number_of_and_input))]
    and1_bias_vector = [0.0 for i in range(len(number_of_and_input))]

    predict_layer1_value = [0.0 for i in range(len(number_of_layer1_neuron))]
    predict_and_value = [0.0 for i in range(len(number_of_and_input))]

    sum_error = 0
    for i_epoch in range(n_epoch):
        for data, validate in x_in, target:
            # for out_put_number in range(len(target)):
            #     predict_layer1_value = []
            #     sum_error = 0.0
            for neuron_layer1_number in range(number_of_layer1_neuron):
                activation_layer1 = find_net(data,
                                             input_weights_vector[neuron_layer1_number],
                                             input_bias_vector[neuron_layer1_number])
                predict_layer1_value[neuron_layer1_number] = predict_step(activation_layer1)

            for and_number in range(number_of_and_input):
                activation_and = find_net(predict_layer1_value,
                                          and1_weights_vector[and_number],
                                          and1_bias_vector[and_number])
                predict_and_value[and_number] = predict_step(activation_and)

                error = validate[and_number] - predict_and_value[and_number]
                if error != 0 and validate[and_number] == 1:
                    for neuron_layer1_number in range(number_of_layer1_neuron):
                        if predict_layer1_value[neuron_layer1_number] < 0:
                            input_weights_vector[neuron_layer1_number] = input_weights_vector[neuron_layer1_number] + l_rate *

            sum_error += error ** 2
            if sum_error != 0:
                bias_vector[outPutNumber] = bias_vector[outPutNumber] + l_rate * row[64 + outPutNumber]
                for j in range(63):
                    weights_vector[outPutNumber][j] = weights_vector[outPutNumber][j] + l_rate * row[
                        64 + outPutNumber] * row[j]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (i_epoch, l_rate, sum_error))
    return weights_vector, bias_vector


epoch = 4
weights, bias = train_weights_perception(dataset, 0.2, epoch)
print(weights[0])
print(bias)
###############          Enter your code above ...           ##################

print("\nThe Neural Network has been trained in " + str(epoch) + " epochs.")

###############                   Testing                    ##################
test_dataset = np.array(read_train_file("OCR_test.txt"))
