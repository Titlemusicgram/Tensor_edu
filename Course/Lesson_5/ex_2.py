import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# Градиентный спуск с мини-батчами
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

for i in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = y[n_batch * BATCH_SIZE : (n_batch + 1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch + 1) * BATCH_SIZE]

        with tf.GradientTape() as tape:
            y_pred = k * x_batch + b
            loss = tf.reduce_mean(tf.square(y_batch - y_pred))

        dk, db = tape.gradient(loss, [k, b])

        k.assign_sub(learning_rate * dk)
        b.assign_sub(learning_rate * db)

print(k, b, sep='\n')

y_pr = k * x + b
plt.scatter(x, y, s=0.2)
plt.scatter(x, y_pr, s=2, c='red')
plt.show()
