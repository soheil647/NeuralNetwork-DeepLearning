# Neural Networks
# Bonus Assignment
import numpy as np


##############################      Main      #################################
def read_train_file(file="OCR_train.txt"):
    training_data_list = []
    train_file = open(file, "r")
    for line in train_file:
        line = list(line.replace(" ", ""))
        line = [int(x) * 2 - 1 for x in line if x != "\n"]
        training_data_list.extend([line[:]])
    return training_data_list


###############                 Training                     ##################


###############          Enter your code below ...           ##################
dataset = np.array(read_train_file())
# dic = {'name': np.array([item[:63] for item in dataset]).reshape([21, 9, 7]), 'id': [item[63:] for item in dataset]}
# dic = {'name': np.array([item[:63] for item in dataset]), 'id': [item[64:] for item in dataset]}
# print(dic["id"][0])


# Make a prediction with weights
def predict_h(row, out_weight, out_bias):
    activation = out_bias
    for j in range(63):
        activation += out_weight[j] * row[j]
    return 1.0 if activation >= 0.0 else -1.0


# Estimate Perceptron weights using stochastic gradient descent
def train_weights_perception(train, l_rate, n_epoch):
    weights_vector = [[0.0 for i in range(63)] for j in range(7)]
    bias_vector = [0.0 for i in range(7)]
    sum_error = 0
    for i_epoch in range(n_epoch):
        for row in train:
            # print()
            sum_error = 0.0
            for outPutNumber in range(7):
                prediction = predict_h(row, weights_vector[outPutNumber], bias_vector[outPutNumber])
                # print(prediction)
                error = prediction - row[64 + outPutNumber]
                # print(error)
                sum_error += error ** 2
            if sum_error != 0:
                for outPutNumber in range(7):
                    bias_vector[outPutNumber] = bias_vector[outPutNumber] + l_rate * row[64 + outPutNumber]
                    for j in range(63):
                        weights_vector[outPutNumber][j] = weights_vector[outPutNumber][j] + l_rate * row[64 + outPutNumber] * row[j]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (i_epoch, l_rate, sum_error))
    return weights_vector, bias_vector


epoch = 4
weights, bias = train_weights_perception(dataset, 0.4, epoch)
print(weights[0])
print(bias)
###############          Enter your code above ...           ##################

print("\nThe Neural Network has been trained in " + str(epoch) + " epochs.")

###############                   Testing                    ##################


###############          Enter your code below ...           ##################


###############          Enter your code above ...           ##################

# print("\n\nPercent of Error in NN: " + str(_error / _total))
