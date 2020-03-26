# =============================================================================
# Load Data
# =============================================================================

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# import matplotlib.pyplot as plt
# plt.imshow (train_images[1],cmap='binary')

# =============================================================================
#
# preprocess
#
# =============================================================================


print(' train image dimension :', train_images.ndim)
print(' train image  shape :', train_images.shape)
print(' train image  type:', train_images.dtype)

X_train = train_images.reshape(60000, 28, 28, 1)
X_test = test_images.reshape(10000, 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

from keras.utils import np_utils

Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)
# =============================================================================
#
# Create Model
# =============================================================================

from keras.models import Model
from keras import layers

my_input = layers.Input(shape=(28, 28, 1))
conv1 = layers.Conv2D(16, 3, activation='relu', padding='same', strides=1)(my_input)
pool1 = layers.MaxPool2D(pool_size=2)(conv1)
conv2 = layers.Conv2D(32, 3, activation='relu', padding='same', strides=1)(pool1)
pool2 = layers.MaxPool2D(pool_size=2)(conv2)
flat = layers.Flatten()(pool2)
out = layers.Dense(10, activation='softmax')(flat)

myModel = Model(my_input, out)

myModel.summary()

myModel.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# =============================================================================
#
# Train Model (Fitting)
# =============================================================================

import datetime

start = datetime.datetime.now()
trained_model = myModel.fit(X_train, Y_train, batch_size=128, epochs=5, validation_split=0.2)
end = datetime.datetime.now()
Total_time_training = end - start

print('Total_time_training:', Total_time_training)

history = trained_model.history

losses = history['loss']
val_losses = history['val_loss']
ac = history['acc']
val_ac = history['val_acc']

import matplotlib.pyplot as plt

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses)
plt.plot(val_losses)
plt.legend(['loss', 'val_loss'])

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(ac)
plt.plot(val_ac)
plt.legend(['acc', 'val_acc'])

# =============================================================================
#
# Evaluation
# =============================================================================

predicted_labels = myModel.predict(X_test)
# plt.imshow (test_images[0],cmap='binary')
test_loss, test_acc = myModel.evaluate(X_test, Y_test)