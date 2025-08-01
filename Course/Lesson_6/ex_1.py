import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers


# Нейронная сеть для сложения двух чисел
class DenseNN(tf.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name='w')
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name='b')

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)

            self.fl_init = True

        y = x @ self.w + self.b

        return y
    

model = DenseNN(1)


# Обучение нейронной сети
x_train = tf.random.uniform(shape=(100, 2), minval=0, maxval=10)
y_train = [a + b for a, b in x_train]

EPOCHS = 50
learning_rate = 0.01

loss = lambda x, y: tf.reduce_mean(tf.square(x - y))
opt = optimizers.adam_v2.Adam(learning_rate=learning_rate)

for n in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.expand_dims(x, axis=0)
        y = tf.constant(y, shape=(1, 1))
    
        with tf.GradientTape() as tape:
            f_loss = loss(y, model(x))

        grads = tape.gradient(f_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    print(f_loss.numpy())

print(model.trainable_variables)
print(model(tf.constant([[1.0, 2.0]])))
