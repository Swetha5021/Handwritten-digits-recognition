# HANDWRITTEN DIGITS RECOGNITION
# By SWETHA S
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

digit_mnist = keras.datasets.mnist
(X_train_full,y_train_full) , (X_test,y_test) = digit_mnist.load_data()

class_names = ["0 - Zero" , "1 - One" , "2 - Two" , "3 - Three" , "4 - Four" , "5 - Five" , "6 - Six" , "7 - Seven" , "8 - Eight", "9 - Nine"]

X_train_n = X_train_full / 255.0
X_test_n = X_test / 255.0

X_valid , X_train = X_train_n[:6000] , X_train_n[6000:]
y_valid , y_train = y_train_full[:6000] , y_train_full[6000:]
X_test = X_test_n

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(200, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))

model.summary()

import pydot
keras.utils.plot_model(model)

model.compile(loss = "sparse_categorical_crossentropy",
             optimizer = "sgd" , 
             metrics = ["accuracy"])
             
model_history = model.fit(X_train, y_train, epochs=60,
                         validation_data = (X_valid,y_valid))
                 
pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.evaluate(X_test, y_test)

X_new = X_test[:3]

y_pred = np.argmax(model.predict(X_test), axis=-1)
y_pred

print(plt.imshow(X_test[0]))
np.array(class_names)[y_pred][0]
