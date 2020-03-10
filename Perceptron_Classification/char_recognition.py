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
print(np.array(dataset).shape)
# dic = {'name': np.array([item[:63] for item in dataset]).reshape([21, 9, 7]), 'id': [item[63:] for item in dataset]}
dic = {'name': np.array([item[:63] for item in dataset]), 'id': [item[64:] for item in dataset]}
print(len(dic['name'][0]))
print(len(dic['id'][0]))


# Make a prediction with weights
def predict_h(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else -1.0


# Estimate Perceptron weights using stochastic gradient descent
def train_weights_perception(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict_h(row, weights)
            error = row[-1] - prediction
            if error != 0:
                # print(">epoch=%d, prediction=%d, error=%d, expected=%d" % (epoch, prediction, error, row[-1]))
                sum_error += error ** 2
                weights[0] = weights[0] + l_rate * row[-1]
                for i in range(len(row) - 1):
                    weights[i + 1] = weights[i + 1] + l_rate * row[-1] * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights


###############          Enter your code above ...           ##################

print("\nThe Neural Network has been trained in " + str(epoch) + " epochs.")

###############                   Testing                    ##################


###############          Enter your code below ...           ##################


###############          Enter your code above ...           ##################

print("\n\nPercent of Error in NN: " + str(_error / _total))
