import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
from tensorflow.python.keras import optimizers

x = tf.Variable(-1.0)
y = lambda: x ** 2 - x

N = 100
opt = optimizers.gradient_descent_v2.SGD(0.1)

for n in range(N):
    opt.minimize(y, [x])

print(x.numpy())
