import numpy as np
import matplotlib.pyplot as plt


# First Dataset 200 Point Equal 100 vs 100
    # Uncomment For First Part
# valuesOfPositive = 1 + 0.5 * np.random.normal(0, 1, (2, 100))
# valuesOfNegative = -1 + 0.5 * np.random.normal(0, 1, (2, 100))

# Second Dataset 110 point 10 vs 100
    # Uncomment For Second Part
valuesOfPositive = 1 + 0.5 * np.random.normal(0, 1, (2, 100))
valuesOfNegative = -1 + 0.5 * np.random.normal(0, 1, (2, 10))

plt.plot(valuesOfPositive[0], valuesOfPositive[1], "bo")
plt.plot(valuesOfNegative[0], valuesOfNegative[1], "ro")
plt.grid()
# plt.show()
dataset = []
for j in range(len(valuesOfPositive[0])):
    dataset.append([valuesOfPositive[0][j], valuesOfPositive[1][j], 1])

for j in range(len(valuesOfNegative[0])):
    dataset.append([valuesOfNegative[0][j], valuesOfNegative[1][j], -1])


# print(dataset[0][2])
# print(np.array(dataset).reshape(200, 3))


# Make a prediction with weights
def predict_h(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else -1.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights_Preceptron(train, l_rate, n_epoch):
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


def predict_net(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return activation

def train_weights_Adaline(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict_net(row, weights)
            error = row[-1] - prediction
            if error != 0:
                # print(">epoch=%d, prediction=%d, error=%d, expected=%d" % (epoch, prediction, error, row[-1]))
                sum_error += error
                weights[0] = weights[0] + l_rate * error
                for i in range(len(row) - 1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights



l_rate = 0.6
n_epoch = 5
# weights = train_weights_Preceptron(dataset, l_rate, n_epoch)
weights = train_weights_Adaline(dataset, l_rate, n_epoch)
print(weights)
x = []
y = []
for i in range(len(dataset)):
    x.append((-weights[2]/weights[1]) * dataset[i][1] - weights[0])
    y.append(dataset[i][1])

plt.plot(y, x)
plt.show()