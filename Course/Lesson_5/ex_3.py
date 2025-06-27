import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers


# Градиентный спуск с мини-батчами и встроенным алгоритмом SGD
TOTAL_POINTS = 1000

x = np.random.uniform(0, 10, size=[TOTAL_POINTS])
noise = np.random.normal(loc=0, scale=0.2, size=[TOTAL_POINTS])

k_true = 0.7
b_true = 2.0

y = k_true * x + b_true + noise

plt.scatter(x=x, y=y, s=0.2)
plt.show()

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 50
learning_rate = 0.02

BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE

opt = optimizers.gradient_descent_v2.SGD(learning_rate=learning_rate) # обычный градиентный спуск
# opt = optimizers.gradient_descent_v2.SGD(learning_rate=learning_rate, momentum=0.5) # метод моментов
# opt = optimizers.gradient_descent_v2.SGD(learning_rate=learning_rate, momentum=0.5, nesterov=True) # моменты Нестерова
# opt = optimizers.adagrad_v2.Adagrad(learning_rate=0.2) # Adagrad (заметь, что с 0.02 работает плохо, в с 0.2 нормально)
# opt = optimizers.adadelta_v2.Adadelta(learning_rate=4.0) # Adadelta (заметь, что даже с 4.0 работает не очень. плохой алгоритм для градиентного спуска)
# opt = optimizers.rmsprop_v2.RMSprop(learning_rate=learning_rate) # RMSProp
# opt = optimizers.adam_v2.Adam(learning_rate=0.1) # Adam. Считается лучшим для старта подбора. Заметь, что 0.1 лучше, чем 0.01

for i in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = y[n_batch * BATCH_SIZE : (n_batch + 1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch + 1) * BATCH_SIZE]

        with tf.GradientTape() as tape:
            y_pred = k * x_batch + b
            loss = tf.reduce_mean(tf.square(y_batch - y_pred))

        dk, db = tape.gradient(loss, [k, b])

        opt.apply_gradients(zip([dk, db], [k, b]))

print(k, b, sep='\n')

y_pr = k * x + b

plt.scatter(x, y, s=0.2)
plt.scatter(x, y_pr, s=2, c='red')
plt.show()
