import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random

from tensorflow.python.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential

my_model = Sequential()
my_model.add(Conv2D(8, ))

def load_data(dataset, csv):
    images = []
    classes = []
    rows = pd.read_csv(dataset + csv)
    rows = rows.sample(frac=1).reset_index(drop=True)

    for i, row in rows.iterrows():
        img_class = row["ClassId"]
        img_path = row["Path"]
        image = os.path.join(dataset, img_path)
        image = cv2.imread(image)
        image_rs = cv2.resize(image, (30, 30), 3)

        R, G, B = cv2.split(image_rs)

        img_r = cv2.equalizeHist(R)
        img_g = cv2.equalizeHist(G)
        img_b = cv2.equalizeHist(B)

        new_image = cv2.merge((img_r, img_g, img_b))

        if i % 500 == 0:
            print(f"loaded: {i}")
        images.append(new_image)
        classes.append(img_class)

    X = np.array(images)
    y = np.array(images)
    return X, y


train_data = r"/home/sspc/Desktop/gtsrb-german-traffic-sign"
test_data = r"/home/sspc/Desktop/gtsrb-german-traffic-sign"
(train_X, train_Y) = load_data(train_data, "/Train.csv")
(test_X, test_Y) = load_data(test_data, "/Test.csv")

print("UPDATE: Normalizing data")
train_X = train_X.astype("float64") / 255.0
test_X = test_X.astype("float64") / 255.0
print("UPDATE: One-Hot Encoding data")
num_labels = len(np.unique(train_Y))
trainY = to_categorical(train_Y, num_labels)
testY = to_categorical(test_Y, num_labels)

class_totals = trainY.sum(axis=0)
class_weight = class_totals.max() / class_totals
