
# =============================================================================
#
# imports
#
# =============================================================================


import pandas as pd
import matplotlib.pyplot as plt
import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu
from keras.optimizers import adam
from keras.losses import mean_squared_logarithmic_error


def one_layer_network_prediction():
    # =============================================================================
    #
    # reading data
    #
    # =============================================================================

    dataset = pd.read_csv("/home/sspc/Desktop/Neural Networks/MLP_Madaline_DimensionalityReduction/House Sales.csv")
    dataset = dataset.iloc[:5000]

    x_train = dataset.values[:4000, 2:]
    y_train = dataset.values[:4000, 2]

    x_test = dataset.values[4000:, 2:]
    y_test = dataset.values[4000:, 2]
    print(x_test)
    # =============================================================================
    #
    # Model
    #
    # =============================================================================

    my_model = Sequential()

    my_model.add(Dense(units=19, activation=relu, input_shape=(19,)))
    my_model.add(Dense(units=1))

    my_model.compile(optimizer="Adam", loss='MSE', metrics=['mean_squared_logarithmic_error'])

    my_train = my_model.fit(x=x_train, y=y_train, batch_size=32, epochs=200, validation_split=0.2)

    # =============================================================================
    #
    # test results
    #
    # =============================================================================

    history = my_train.history
    test_loss, test_acc = my_model.evaluate(x=x_test, y=y_test)

    predicted_labels = my_model.predict(x_test)

    print(test_loss)
    print(test_acc)
    print(predicted_labels)

    print(history)
    print(history["loss"])
    print(history["val_loss"])

    plt.plot(history["loss"][5:])
    plt.plot(history["val_loss"][5:])
    plt.legend(["loss", "val_loss"])
    plt.show()

    # plt.legend(["acc", "val_acc"])
    # plt.plot(history["accuracy"][5:])
    # plt.plot(history["val_accuracy"][5:])
    # plt.show()


def two_layer_network_prediction():
    # =============================================================================
    #
    # reading data
    #
    # =============================================================================

    dataset = pd.read_csv("/home/sspc/Desktop/Neural Networks/MLP_Madaline_DimensionalityReduction/House Sales.csv")
    dataset = dataset.iloc[:5000]

    x_train = dataset.values[:4000, 2:]
    y_train = dataset.values[:4000, 2]

    x_test = dataset.values[4000:, 2:]
    y_test = dataset.values[4000:, 2]
    print(x_test)
    # =============================================================================
    #
    # Model
    #
    # =============================================================================

    my_model = Sequential()

    my_model.add(Dense(units=19, activation='relu', input_shape=(19,)))
    my_model.add(Dense(units=19, activation='relu'))
    my_model.add(Dense(units=1,))

    my_model.compile(optimizer="Adam", loss='MSE', metrics=['mean_squared_logarithmic_error'])

    my_train = my_model.fit(x=x_train, y=y_train, epochs=200, validation_split=0.2, )

    # =============================================================================
    #
    # test results
    #
    # =============================================================================

    history = my_train.history
    test_loss, test_acc = my_model.evaluate(x=x_test, y=y_test)

    predicted_labels = my_model.predict(x_test)

    print(test_loss)
    print(test_acc)
    print(predicted_labels)

    print(history)
    print(history["loss"])
    print(history["val_loss"])

    plt.plot(history["loss"][5:])
    plt.plot(history["val_loss"][5:])
    plt.legend(["loss", "val_loss"])
    plt.show()


# =============================================================================
#
# MLP with 1 hidden layer
#
# =============================================================================

# start = datetime.datetime.now()
# one_layer_network_prediction()
# end = datetime.datetime.now()
# print(end - start)

# =============================================================================
#
# MLP with 2 hidden layer
#
# =============================================================================

start = datetime.datetime.now()
two_layer_network_prediction()
end = datetime.datetime.now()
print(end - start)

# plt.legend(["1 layer loss", "2 layer loss"])
# plt.show()