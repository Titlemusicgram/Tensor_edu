import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Создаем игрушечную выборку данных
N = 1000
k_true = 0.7
b_true = 2.0
x = np.random.uniform(low=0, high=10, size=[N])
noise = np.random.normal(loc=0, scale=0.2, size=[N])
y = k_true * x + b_true + noise

# plt.scatter(x=x, y=y, s=2)
# plt.show()


# Тренируем модель
k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 500
learning_rate = 0.02

opt = tf.keras.optimizers.SGD(learning_rate)

for n in range(EPOCHS):
    with tf.GradientTape() as tape:
        f = k * x + b
        loss = tf.reduce_mean(tf.square(y-f))

    dk, db = tape.gradient(loss, [k, b])

    opt.apply_gradients(zip([dk, db], [k, b]))

print(k.numpy(), b.numpy(), sep='\n')

y_pr = k * x + b
plt.scatter(x, y, s=2)
plt.scatter(x, y_pr, c='r', s=2)
plt.show()
