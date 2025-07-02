import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Input


# Подготовка данных
N = 1000

x_min = 0.0
x_max = 10.0

x_train = np.random.uniform(x_min, x_max, size=(N, 2)).tolist()
y_train = [i + j for i, j in x_train]

x_test = np.random.uniform(x_min, x_max, size=(N//5, 2)).tolist()
y_test = [i + j for i, j in x_test]

x_train = tf.Variable(x_train)
y_train = tf.Variable(y_train)

x_test = tf.Variable(x_test)
y_test = tf.Variable(y_test)


# Модель
model = Sequential([
    Input(shape=[2,]),
    Dense(1, activation=keras.activations.linear)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='mean_squared_error',
    metrics=[keras.metrics.MeanAbsoluteError]
)

model.fit(x_train, y_train, batch_size=20, epochs=50, validation_batch_size=0.2)

print('\n\n')
print(model.evaluate(x_test, y_test, batch_size=20), end='\n\n\n')

print(model.predict(tf.constant([[5.0, 6.0]])))
