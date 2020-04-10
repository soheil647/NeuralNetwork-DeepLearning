import numpy as np
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.activations import softmax
from keras.optimizers import adam
from keras.losses import sparse_categorical_crossentropy

from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import PCA


def NN_with(x_train, y_train, nouron_number_layer1, nouron_number_layer2, batch_size, epoch):
    my_model = Sequential()
    my_model.add(Dense(nouron_number_layer1, activation='relu', input_shape=(128,)))
    my_model.add(Dense(nouron_number_layer2, activation='relu'))
    my_model.add(Dropout(0.3))
    my_model.add(Dense(10, activation=softmax))
    my_model.compile(optimizer='sgd', loss=sparse_categorical_crossentropy, metrics=['accuracy'])
    trained_model = my_model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_split=0.2)
    return trained_model, my_model

def NN_with_rbm(x_train, x_test):
    my_model = BernoulliRBM(n_components=128)
    rbm_fit = my_model.fit(x_train)
    x_train = my_model.transform(x_train)
    x_test = my_model.transform(x_test)
    return x_train, x_test

def NN_with_PCA(x_train, x_test):
    pca = PCA(n_components=128)
    pca_fit = pca.fit_transform(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    return x_train , x_test


def my_nn(nouron_number_layer1, nouron_number_layer2, batch_size, epoch, method):
    print("\n\n\n Network With layer1 of: ", nouron_number_layer1, "Nourons and layer2 of: ", nouron_number_layer2, "and Batch_size: ", batch_size, "And Epoch: ", epoch)
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    x_train = np.array(x_train).reshape(60000, 28 * 28)
    x_test = np.array(x_test).reshape(10000, 28 * 28)

    start_time = time.time()
    if method == "PCA":
        x_train, x_test = NN_with_PCA(x_train, x_test)
    if method == "RBM":
        x_train, x_test = NN_with_rbm(x_train, x_test)
    trained_model, my_model = NN_with(x_train, y_train, nouron_number_layer1, nouron_number_layer2, batch_size, epoch)
    print("Trained finished in: ", time.time() - start_time)
    history = trained_model.history
    test_loss, test_acc = my_model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

    plt.title('NN with layer1 of: ' + str(nouron_number_layer1) + ' layer2 of: ' + str(nouron_number_layer2) + ' batch_size of: ' + str(batch_size))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['loss', 'val_loss'])

    plt.figure()
    plt.title('NN with layer1 of: ' + str(nouron_number_layer1) + ' layer2 of: ' + str(nouron_number_layer2) + ' batch_size of: ' + str(batch_size))
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.legend(['acc', 'val_acc'])

    plt.show()

    from sklearn.metrics import confusion_matrix
    test_prediction = my_model.predict_classes(x_test)
    matrix = confusion_matrix(y_true=y_test, y_pred=test_prediction)
    print("Matrix= ")

    print(matrix)

    marks = np.arange(len(class_names))
    cmap = plt.cm.Blues
    # title = 'Confusion matrix'
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title('NN with layer1 of: ' + str(nouron_number_layer1) + ' layer2 of: ' + str(nouron_number_layer2) + ' batch_size of: ' + str(batch_size) + ' For Confusion matrix')
    plt.colorbar()
    plt.xticks(marks, class_names, rotation=45)
    plt.yticks(marks, class_names)
    plt.tight_layout()
    plt.show()


my_nn(128, 30, 64, 30, "PCA")
my_nn(128, 30, 64, 30, "RBM")
my_nn(128, 30, 64, 30, "PCA")
