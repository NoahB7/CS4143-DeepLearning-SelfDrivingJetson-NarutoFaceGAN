from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import time
import sys

pre = time.time()

df = pd.read_csv(sys.argv[2])
Xpaths = df.iloc[0:50000,0:1]
Y = df.iloc[0:50000,1:2]
encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(Y)
X = []


for path in Xpaths['Files']:
    img = cv2.imread(sys.argv[1] + "/" + path,0)/255.0
    img = cv2.resize(img,(60,40))
    X.append(img)
X = np.asarray(X)
X = X.reshape( len(X), 40, 60, 1)


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)


model = Sequential()

model.add(Input(shape = (40,60,1), dtype="float32"))

model.add(Conv2D(16,kernel_size=(5,5), strides=3, use_bias=True, activation='relu'))

model.add(MaxPooling2D(pool_size=(3,3), strides=2))

model.add(Conv2D(16,kernel_size=(1,1), strides=1, use_bias=True, activation='relu'))
model.add(Conv2D(32,kernel_size=(1,1), strides=1, use_bias=True, activation='relu'))
model.add(Conv2D(32,kernel_size=(3,3), strides=1, use_bias=True, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=1))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(3, activation='linear'))

model.summary()


model.compile( optimizer="adam" , loss="mse", metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

post = time.time()

evaluation = model.evaluate(x_test,y_test)

print(evaluation)
print(post-pre)