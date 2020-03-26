import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.activations import softmax
from keras.losses import sparse_categorical_crossentropy

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train = x_train / 255.0
x_test = x_test / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[y_train[i]])
# plt.show()

my_model = Sequential()

my_model.add(Flatten(input_shape=(28, 28)))
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

# matrix = tf.math.confusion_matrix(class_names, my_model.predict(x_test))
# print(matrix)