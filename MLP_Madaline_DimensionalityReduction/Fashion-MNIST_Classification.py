import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.activations import softmax
from keras.losses import sparse_categorical_crossentropy

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[y_train[i]])
# plt.show()

x_train = np.array(x_train).reshape(60000, 28*28)
x_test = np.array(x_test).reshape(10000, 28*28)

# print(np.array(x_train).shape)

my_model = Sequential()
my_model.add(Dense(128, activation='relu'))
my_model.add(Dense(10, activation=softmax))

my_model.compile(optimizer='adam', loss=sparse_categorical_crossentropy, metrics=['accuracy'])

trained_model = my_model.fit(x_train, y_train, batch_size=20, epochs=10, validation_split=0.2)


history = trained_model.history

test_loss, test_acc = my_model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.legend(['loss', 'val_loss'])

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.legend(['acc', 'val_acc'])

plt.show()

from sklearn.metrics import confusion_matrix
test_prediction = my_model.predict_classes(x_test)
matrix = confusion_matrix(y_true=y_test, y_pred=test_prediction)
print("Matrix= ", matrix)
marks = np.arange(len(class_names))
cmap = plt.cm.Blues
title='Confusion matrix'
plt.figure()
plt.imshow(matrix, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
plt.xticks(marks, class_names, rotation=45)
plt.yticks(marks, class_names)
plt.tight_layout()
plt.show()